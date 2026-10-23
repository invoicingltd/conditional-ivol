"""Run one version in a fresh process; all observations are invented here.

Invoked by test_legacy_equivalence.py, never imports the two implementations
into one process, and writes results only to pytest's temporary directory.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np


def scientific_values(value):
    """Discard elapsed timing only; preserve all scientific results."""
    if isinstance(value, dict):
        return {str(k): scientific_values(v) for k, v in value.items() if k != "timings"}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [scientific_values(v) for v in value]
    return value


def modules(args):
    if args.implementation == "legacy":
        sys.path[:0] = [
            str(args.legacy_root / "thesis_method" / "core"),
            str(args.legacy_root / "thesis_method" / "rank_rule"),
            str(args.legacy_root / "src"),
        ]
        prefix = ""
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        prefix = "civol.core."
    core = {name: importlib.import_module(prefix + name)
            for name in ("common_space", "estimator", "horizon", "outcomes")}
    rank_name = "rank_rules" if args.implementation == "legacy" else "civol.rank_rule.rank_rules"
    core["rank_rules"] = importlib.import_module(rank_name)
    return core


def panel_results(m):
    rng = np.random.default_rng(20261022)
    T, N, K, H = 120, 24, 2, 5
    factors = rng.normal(0, .01, (T + H, K))
    loadings = rng.normal(.5, .3, (N, K))
    x = .0002 + factors @ loadings.T + rng.normal(0, .012, (T + H, N))
    ff = np.column_stack([factors, rng.normal(0, .008, T + H)])
    rf = np.full(T + H, .0001)
    config = dict(m["estimator"].CONFIG, T=T, H=H, pairs=128, batches=4,
                  kmax=5, sign_draws=19, alternative_ranks=(2, 4),
                  variant_rank=3, ff3_long=60, backend="numpy")
    estimate = m["estimator"].estimate_panel(
        x[:T], ff[:T], rf[:T], seed=213, config=config, simple=np.expm1(x[:T]))
    valid = np.ones((H, N), dtype=bool)
    valid[1, 2] = False
    issuer = estimate["issuer"]
    outcomes = m["outcomes"].realized_outcomes(
        x[T:], valid, ff[T:], rf[T:], issuer["mu"], issuer["s"],
        estimate["B_std"], issuer["ff3_alpha"], issuer["ff3_betas"], H,
        alternative_B_std=estimate["alternative_B_std"])
    cs, rr = m["common_space"], m["rank_rules"]
    _, s, Z = cs.standardize(x[:T])
    eig = cs.gram_eigen(Z)
    signs = rr.balanced_signs(N, 16, 711)
    insample = rr.insample_curves(x[:T], kmax=5, signs=signs)
    heldout = rr.heldout_curve(x[:T], split=80, kmax=5)
    rank = dict(insample=insample, heldout=heldout,
                rule_a=rr.rule_a(insample, 1.25), rule_b=rr.rule_b(heldout, 1.25),
                decomposition=rr.decomposition_at(Z, eig, s, 3),
                next_direction=rr.next_direction(Z, eig, s, 3))
    return dict(estimate=estimate, outcomes=outcomes, rank=rank)


def gpu_results(m, gpu_python):
    # Both versions must run the identical requested interpreter and report GPU.
    m["horizon"].GPU_PYTHON = gpu_python
    rng = np.random.default_rng(20261023)
    N, K = 12, 2
    b = rng.normal(.004, .006, (N, K))
    d = np.exp(rng.normal(np.log(.0002), .2, N))
    model = dict(mu=np.full(N, .0002), loadings=b, d=d,
                 reference=np.sum(b * b, axis=1) + d,
                 h_next=np.linspace(.5, 1.5, N), m_next=np.linspace(.8, 1.2, N),
                 coefficients=np.array([.06, .85, .04, .995, .02]),
                 factor_omega=np.array([.05, .06]),
                 factor_alpha=np.array([.05, .08]),
                 factor_beta=np.array([.9, .86]), g_next=np.array([1.2, .9]))
    results = {}
    for drivers in (("total", "residual"), ("total", "total"),
                    ("residual", "residual"), ("residual", "total")):
        result = m["horizon"].paired_moments(
            dict(model, drivers=drivers), 5, pairs=256, batches=4, seed=317,
            backend="gpu")
        if result["backend"] != "gpu":
            raise AssertionError(f"GPU fallback: {result.get('gpu_error')}")
        results["/".join(drivers)] = result
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "package"), required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("panel", "gpu"), required=True)
    parser.add_argument("--gpu-python", type=Path)
    args = parser.parse_args()
    m = modules(args)
    result = panel_results(m) if args.mode == "panel" else gpu_results(m, args.gpu_python)
    args.output.write_text(json.dumps(scientific_values(result)), encoding="utf-8")


if __name__ == "__main__":
    main()
