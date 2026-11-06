"""Versioned cIVOL implementation with a Bai--Ng safeguard on the EW rank.

The only numerical change from the legacy rule-A estimator is the selected
rank: max(K_EW, K_BaiNg).  Fitting, final-state convention, bounded factor law,
paired integration and the rank-dependent seed are delegated unchanged to
``rank_rule.estimate_rank_rule.civol_at_rank``.  Legacy workflows and their
record directories remain available for exact replication.

The safeguard prevents the EW criterion from selecting fewer directions than
Bai--Ng; it does not guarantee recovery of every common payoff response or
general residual diversifiability.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from civol.core.benchmarks import ff3_space
from civol.core.common_space import bai_ng_icp2, gram_eigen, residual_common_content, standardize
from civol.core.outcomes import realized_outcomes
from civol.core.paths import CACHE, PACKAGE, records_path
from civol.core.universe import eligible_universe
from civol.rank_rule import rank_rules
from civol.rank_rule.estimate_rank_rule import civol_at_rank, source_hashes as legacy_source_hashes
from civol.records import completed_record_stems


VERSION = "safeguarded_measurement_v1"
RECORDS = records_path("safeguarded_measurement")
CONFIG = dict(T=750, H=21, pairs=16384, batches=8, bound=1.25,
              ew_kmax=40, bai_ng_kmax=20, transitory_driver="residual",
              level_driver="total", factor_bound_multiple=25.,
              seed_offset=100, backend="numpy")


def configuration(config=None):
    """Validate an explicitly versioned method; numerical budgets may vary."""
    result = dict(CONFIG)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        result.update(config)
    fixed = ("bound", "ew_kmax", "bai_ng_kmax", "transitory_driver",
             "level_driver", "factor_bound_multiple", "seed_offset")
    for name in fixed:
        if result[name] != CONFIG[name]:
            raise ValueError(f"Changing {name} would change {VERSION}")
    for name in ("T", "H", "pairs", "batches"):
        value = result[name]
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        result[name] = int(value)
    if result["T"] <= 12:
        raise ValueError("T must exceed twelve sessions")
    if result["batches"] < 2 or result["pairs"] % result["batches"]:
        raise ValueError("Pairs must divide into at least two batches")
    if result["backend"] not in ("numpy", "gpu"):
        raise ValueError("Backend must be numpy or gpu")
    return result


def safeguarded_rank(ew_rank, bai_ng_rank):
    """Combine formation-only ranks without inspecting outcomes or losses."""
    for value in (ew_rank, bai_ng_rank):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 0:
            raise ValueError("Ranks must be nonnegative integers")
    return max(int(ew_rank), int(bai_ng_rank))


def select_common_space(x, config=None):
    """Compute both original selectors once on the same formation panel."""
    cfg = configuration(config)
    x = np.asarray(x, dtype=np.float64)
    mu, s, Z = standardize(x)
    T, N = Z.shape
    if T != cfg["T"] or N <= 12:
        raise ValueError("Panel must have T configured rows and over twelve issuers")
    eig = gram_eigen(Z)
    curves = rank_rules.insample_curves(x, cfg["ew_kmax"], prepared=(s, Z, eig))
    ew_rank, attainable = rank_rules.rule_a(curves, cfg["bound"])
    bn_rank, criteria = bai_ng_icp2(eig[0], T, N, cfg["bai_ng_kmax"])
    rank = safeguarded_rank(ew_rank, bn_rank)
    if rank < len(curves["equal_weight"]):
        ew_selected = float(curves["equal_weight"][rank])
    else:
        # In very small panels Bai--Ng can reach N-1 while the original
        # EW search ends at N-2.  Preserve both original search rules.
        residuals = rank_rules.decomposition_at(Z, eig, s, rank)["residuals"]
        ew_selected = rank_rules.equal_weight_ratio(residuals)
    ew_floor = float(min(curves["equal_weight"]))
    ew_threshold = float(cfg["bound"] * max(1., ew_floor))
    diagnostics = dict(
        rule="max(equal_weight, bai_ng_icp2)", rank=rank,
        ew_rank=int(ew_rank), bai_ng_rank=int(bn_rank),
        safeguard_changed_rank=bool(rank != ew_rank),
        ew_bound=float(cfg["bound"]), ew_search_max=int(curves["kmax"]),
        bai_ng_search_max=int(min(cfg["bai_ng_kmax"], N - 1, T - 3)),
        ew_floor=ew_floor, ew_selection_threshold=ew_threshold,
        ew_absolute_bound_attainable=bool(attainable),
        selected_ew_ratio=ew_selected,
        selected_ew_absolute_bound_satisfied=bool(ew_selected <= cfg["bound"]),
        selected_ew_selection_threshold_satisfied=bool(ew_selected <= ew_threshold),
        ew_curve=curves["equal_weight"], bai_ng_criteria=criteria,
        top_eigenvalues=eig[0][:41].tolist(),
    )
    return dict(x=x, mu=mu, s=s, Z=Z, eig=eig, rank=rank,
                diagnostics=diagnostics, config=cfg)


def estimate_panel(x, base_seed, config=None):
    """Fit the safeguarded method without changing the legacy at-rank kernel."""
    prepared = select_common_space(x, config)
    cfg, rank = prepared["config"], prepared["rank"]
    seed = int(base_seed) + cfg["seed_offset"] + rank
    forecasts, model_summary, B = civol_at_rank(
        prepared["x"], prepared["Z"], prepared["eig"], prepared["s"],
        prepared["mu"], rank, seed, cfg)
    summary = dict(version=VERSION, T=cfg["T"], N=int(prepared["x"].shape[1]),
                   rank=rank, rank_diagnostics=prepared["diagnostics"],
                   base_seed=int(base_seed), integration_seed=seed,
                   config=cfg, model=model_summary)
    return dict(forecasts=forecasts, summary=summary, B_std=B,
                mu=prepared["mu"], s=prepared["s"])


def source_hashes():
    result = legacy_source_hashes()
    result[str(Path(__file__).relative_to(PACKAGE))] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    return result


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(str(type(value)))


def run_date(cache, t_idx, config=None, out=RECORDS):
    """Write new-version records only, refusing completed or partial targets."""
    cfg = configuration(config)
    started = time.perf_counter()
    date = str(cache.dates[t_idx])
    out = Path(out)
    stem = out / date
    if any(Path(str(stem) + suffix).exists() for suffix in (".json", ".npz", ".json.tmp", ".npz.tmp")):
        raise FileExistsError(f"Refusing to replace an existing or partial record: {stem}")
    T, H = cfg["T"], cfg["H"]
    idx, universe = eligible_universe(cache, t_idx, T)
    rows = slice(t_idx - T + 1, t_idx + 1)
    x = np.asarray(cache.ret[rows][:, idx], dtype=np.float64)
    ff = np.asarray(cache.ff[rows], dtype=np.float64)
    base_seed = int(np.datetime64(cache.dates[t_idx], "D").astype(np.int64))
    estimate = estimate_panel(x, base_seed, cfg)
    ffs = ff3_space(x, ff[:, :3], ff[:, 3])
    arrays = dict(estimate["forecasts"], permno=cache.permnos[idx],
                  permco=np.asarray(cache.permco[t_idx])[idx],
                  mktcap=np.asarray(cache.mktcap[t_idx], dtype=np.float64)[idx])
    summary = estimate["summary"]
    summary.update(date=date, t_idx=int(t_idx), universe=universe,
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   source_hashes=source_hashes(),
                   ff3_insample=residual_common_content(ffs["residuals"]))
    available = t_idx + H < len(cache)
    summary["outcome_available"] = bool(available)
    if available:
        future = slice(t_idx + 1, t_idx + H + 1)
        future_ff = np.asarray(cache.ff[future], dtype=np.float64)
        outcome = realized_outcomes(
            np.asarray(cache.ret[future][:, idx], dtype=np.float64),
            np.asarray(cache.valid[future][:, idx]), future_ff[:, :3], future_ff[:, 3],
            estimate["mu"], estimate["s"], estimate["B_std"],
            ffs["alpha"], ffs["betas"], H)
        for name in ("rv_ff3", "rv_ff3_fixed", "rv_broad", "rv_total", "scored"):
            arrays[name] = outcome[name]
        summary["outcomes"] = {name: outcome[name] for name in (
            "scored_count", "diversifiability_ff3_fixed", "diversifiability_broad",
            "diversifiability_total", "portfolio_rv_ff3_fixed", "portfolio_rv_broad")}
    summary["seconds"] = round(time.perf_counter() - started, 3)
    out.mkdir(parents=True, exist_ok=True)
    with Path(str(stem) + ".npz.tmp").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    Path(str(stem) + ".json.tmp").write_text(json.dumps(summary, indent=2, default=_plain), encoding="utf-8")
    Path(str(stem) + ".npz.tmp").replace(Path(str(stem) + ".npz"))
    Path(str(stem) + ".json.tmp").replace(Path(str(stem) + ".json"))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1988-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--limit", type=int, default=0, help="Maximum new dates (0 = all)")
    parser.add_argument("--backend", choices=("numpy", "gpu"), default="numpy")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1:
        parser.error("limit must be nonnegative and threads must be positive")
    from threadpoolctl import threadpool_limits
    from civol.core.universe import Cache
    from civol.application.run_application import schedule
    cfg = configuration(dict(backend=args.backend))
    dates = np.load(CACHE / "dates.npy")
    planned = schedule(dates, cfg["T"], cfg["H"], args.start, args.end)
    done = completed_record_stems(RECORDS)
    for name in done:
        metadata = json.loads((RECORDS / (name + ".json")).read_text(encoding="utf-8"))
        if metadata.get("version") != VERSION or metadata.get("config") != cfg:
            raise ValueError(f"Existing record has a different version or configuration: {name}")
    todo = [index for index in planned if str(dates[index]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    with threadpool_limits(limits=args.threads):
        cache = Cache()
        for index in todo:
            result = run_date(cache, index, cfg)
            print(json.dumps(dict(date=result["date"], rank=result["rank"],
                                  changed=result["rank_diagnostics"]["safeguard_changed_rank"],
                                  seconds=result["seconds"])), flush=True)


if __name__ == "__main__":
    main()
