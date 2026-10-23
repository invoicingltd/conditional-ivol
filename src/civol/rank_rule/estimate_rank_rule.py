"""Estimate cIVOL at the ranks selected by rules A and B and at K = 20.

Uses the core estimation functions and computes out-of-sample diversifiability
diagnostics for one formation date. Each record contains per-issuer NPZ arrays
and a JSON summary; failures are saved as DATE.FAILED.json.
"""
from __future__ import annotations
import hashlib, json, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from civol.core.paths import records_path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
CORE = STAGE / 'core'
from civol.rank_rule import rank_rules as rr
from civol.core.common_space import standardize, gram_eigen, sign_rank, bai_ng_icp2, residual_common_content, SIGN_SEED
from civol.core.private_dynamics import fit_private_dynamics, factor_garch
from civol.core.horizon import paired_moments, surrogate_moments
from civol.core.benchmarks import ff3_space
from civol.core.outcomes import realized_outcomes, broad_residuals
from civol.core.universe import eligible_universe
from civol.core.paths import RECORDS as PRIMARY_RECORDS

VERSION = 'thesis_rank_rule_v1'
RECORDS = records_path('rank_rule')
CONFIG = dict(T=750, H=21, pairs=16384, batches=8, bound=rr.BOUND, kmax=rr.KMAX, split=rr.SPLIT, sign_draws=rr.SIGN_DRAWS,
              fixed_ranks=(20,), diagnostic_ranks=(10, 20), primary_kmax=20, primary_sign_draws=64,
              transitory_driver='residual', level_driver='total', factor_bound_multiple=25., backend='gpu', seed_offset=100)
ROLE_PREFIX = dict(A='civolA', B='civolB', K20='civolK20')
CORE_FILES = ['common_space.py', 'private_dynamics.py', 'horizon.py', 'horizon_gpu.py', 'benchmarks.py', 'outcomes.py', 'universe.py', 'paths.py', 'adaptive_garch.py']
STAGE_FILES = ['rank_rules.py', 'estimate_rank_rule.py']


def source_hashes():
    out = {f'core/{n}': hashlib.sha256((CORE / n).read_bytes()).hexdigest() for n in CORE_FILES}
    out.update({f'rank_rule/{n}': hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in STAGE_FILES})
    return out


def _timer(timings, name, started):
    timings[name] = round(time.perf_counter() - started, 3)
    return time.perf_counter()


def _empty_factor_garch():
    z = np.zeros(0)
    return dict(omega=z, alpha=z, beta=z, g_next=z, target=z, converged=0, optimizer_failures=0, static_selected=0)


def civol_at_rank(x, Z, eig, s, mu, K, seed, config):
    """The cIVOL pipeline at a fixed rank K: common space,
    factor GARCH, pooled two-component private law, paired integration."""
    T, N = Z.shape; H = config['H']
    drivers = (config['transitory_driver'], config['level_driver'])
    dec = rr.decomposition_at(Z, eig, s, K)
    fg = factor_garch(dec['F']) if K else _empty_factor_garch()
    reference = np.sum(dec['loadings'] ** 2, axis=1) + dec['d0']
    z = x - mu
    pdyn = fit_private_dynamics(z, dec['residuals'], dec['d0'], reference, component=True, transitory=drivers[0], level=drivers[1])
    model = dict(mu=mu, loadings=dec['loadings'], d=pdyn['d'], reference=reference, h_next=pdyn['h_next'], m_next=pdyn['m_next'], coefficients=pdyn['coefficients'],
                 drivers=drivers, factor_omega=fg['omega'], factor_alpha=fg['alpha'], factor_beta=fg['beta'], g_next=fg['g_next'], factor_target=fg['target'],
                 factor_bound_multiple=float(config.get('factor_bound_multiple', 25.)))
    pm = paired_moments(model, H, config['pairs'], config['batches'], seed, backend=config.get('backend', 'numpy'))
    sm = surrogate_moments(model, H)
    content = residual_common_content(dec['residuals'])
    forecasts = dict(q=pm['q'], ipv=pm['ipv'], q_mcse=pm['q_mcse'], ipv_mcse=pm['ipv_mcse'], total_variance=pm['total_variance'],
                     surrogate_q=sm['q'], surrogate_ipv=sm['ipv'], d=pdyn['d'], h_next=pdyn['h_next'])
    summary = dict(rank=int(K), seed=int(seed), residual_df_multiplier=T / (T - K - 1), explained_share=float(eig[0][:K].sum() / N) if K else 0.,
                   pooled=dict(coefficients=pdyn['coefficients'].tolist(), objective=pdyn['objective'], converged=pdyn['converged'], static_selected=pdyn['static_selected'],
                               lower_contacts=pdyn['lower_contacts'], upper_contacts=pdyn['upper_contacts'], median_h_next=float(np.median(pdyn['h_next'])),
                               median_m_next=float(np.median(pdyn['m_next']))),
                   factor_garch=dict(alpha=fg['alpha'].tolist(), beta=fg['beta'].tolist(), converged=fg['converged'], optimizer_failures=fg['optimizer_failures'],
                                     static_selected=fg['static_selected']),
                   integration={k: pm.get(k) for k in ('pairs', 'batches', 'backend', 'device', 'gpu_error', 'lower_contacts', 'upper_contacts',
                                                       'common_upper_contacts', 'g_next_clipped_factors', 'factor_bound_multiple')},
                   insample=content)
    summary['integration'].update(median_relative_q_mcse=float(np.median(pm['q_mcse'] / pm['q'])), max_relative_q_mcse=float(np.max(pm['q_mcse'] / pm['q'])),
                                  median_relative_ipv_mcse=float(np.median(pm['ipv_mcse'] / pm['ipv'])), median_compounding_factor=float(np.median(pm['q'] / pm['ipv'])),
                                  median_surrogate_ratio=float(np.median(pm['q'] / sm['q'])), median_private_share=float(np.median(pm['q'] / pm['total_variance'])))
    return forecasts, summary, dec['B']


def estimate_rank_rule_panel(x, ff, rf, base_seed, t_idx, config=CONFIG, roles=None):
    """Rules, forecasts and formation diagnostics for one complete panel.

    roles: optional {role: (rank, seed)} overriding the rules (pilot validation
    only). In production the roles are A (rule A), B (rule B) and K20 with seed
    base_seed + seed_offset + rank, so equal ranks share one integration.
    """
    x = np.asarray(x, float); T, N = x.shape
    timings = {}; t0 = time.perf_counter()
    mu, s, Z = standardize(x)
    w, U = gram_eigen(Z); eig = (w, U)
    sign = sign_rank(Z, w[:config['primary_kmax']], draws=config['primary_sign_draws'], seed=SIGN_SEED)      # reference rank for comparison with application records
    icp2, _ = bai_ng_icp2(w, T, N, config['primary_kmax'])
    signs = rr.balanced_signs(N, config['sign_draws'], rr.SIGN_SEED + int(t_idx))
    ins = rr.insample_curves(x, config['kmax'], signs, prepared=(s, Z, eig))
    ho = rr.heldout_curve(x, config['split'], config['kmax'])
    t0 = _timer(timings, 'rules', t0)
    ka, attainable_a = rr.rule_a(ins, config['bound'])
    kb, attained_b = rr.rule_b(ho, config['bound'])
    sensitivity = dict(rule_a={str(b): rr.rule_a(ins, b)[0] for b in (1.1, 1.25, 1.5, 2.0)}, rule_b={str(b): rr.rule_b(ho, b)[0] for b in (1.1, 1.25, 1.5, 2.0)})
    if roles is None:
        roles = {'A': (ka, base_seed + config['seed_offset'] + ka), 'B': (kb, base_seed + config['seed_offset'] + kb)}
        for k in config['fixed_ranks']:
            roles[f'K{k}'] = (int(k), base_seed + config['seed_offset'] + int(k))
    ffs = ff3_space(x, ff, rf)
    t0 = _timer(timings, 'ff3_space', t0)
    forecasts = {}; models = {}; B_std = {}
    by_rank = {}
    for role, (K, seed) in roles.items():
        key = (int(K), int(seed))
        if key not in by_rank:
            by_rank[key] = civol_at_rank(x, Z, eig, s, mu, int(K), int(seed), config)
            t0 = _timer(timings, f'civol_rank_{K}', t0)
        fc, summ, B = by_rank[key]
        prefix = ROLE_PREFIX.get(role, f'civol{role}')
        forecasts.update({f'{prefix}_{k}': v for k, v in fc.items()})
        models[role] = summ
        B_std[int(K)] = B
    diagnostic_ranks = sorted({int(K) for K, _ in roles.values()} | {int(ka), int(kb), int(icp2), int(sign['rank'])} | set(int(k) for k in config['diagnostic_ranks']))
    for K in diagnostic_ranks:
        if K not in B_std:
            B_std[K] = rr.decomposition_at(Z, eig, s, K)['B']
    next_directions = {K: rr.next_direction(Z, eig, s, K) for K in diagnostic_ranks}
    next_directions['ff3'] = rr.top_direction(ffs['residuals'])
    insample_ff3 = residual_common_content(ffs['residuals'])
    insample_ff3['random_signed'] = rr.random_signed_stats(ffs['residuals'], signs, insample_ff3['equal_weight_residual_variance_ratio'])
    t0 = _timer(timings, 'diagnostic_directions', t0)
    rules = dict(rank_a=int(ka), rank_a_bound_attainable=bool(attainable_a), equal_weight_floor=float(min(ins['equal_weight'])),
                 rank_b=int(kb), rank_b_attained=bool(attained_b), heldout_minimum=float(min(ho['ratio'])), rank_b_issuers_dropped=int(ho.get('dropped_degenerate', 0)),
                 bound=config['bound'], kmax=config['kmax'], split=config['split'], bound_sensitivity=sensitivity,
                 icp2_rank=int(icp2), sign_rank=int(sign['rank']), sign_threshold=sign['threshold'],
                 roles={role: dict(rank=int(K), seed=int(seed)) for role, (K, seed) in roles.items()})
    curves = dict(k=list(range(len(ins['equal_weight']))), insample_equal_weight=ins['equal_weight'], eigenvalues=ins['eigenvalues'],
                  insample_random_signed_mean=ins['random_signed_mean'], insample_random_signed_sd=ins['random_signed_sd'], insample_random_signed_p95=ins['random_signed_p95'],
                  heldout_next_direction=ho['ratio'], heldout_estimation_eigenvalues=ho['estimation_eigenvalues'])
    return dict(forecasts=forecasts, models=models, rules=rules, curves=curves, B_std=B_std, next_directions=next_directions, mu=mu, s=s,
                ff3=dict(alpha=ffs['alpha'], betas=ffs['betas'], insample=insample_ff3), signs=signs, timings=timings, diagnostic_ranks=diagnostic_ranks)


def outcome_diagnostics(future_log, future_valid, ff_future, rf_future, est, H):
    """Realized outcomes (RV proxies, equal-weight diversifiability at every
    diagnostic rank) plus the random-signed and next-direction out-of-sample ratios."""
    mu, s = est['mu'], est['s']
    B_std = est['B_std']; ranks = est['diagnostic_ranks']
    rank_a = est['rules']['rank_a']
    oc = realized_outcomes(future_log, future_valid, ff_future, rf_future, mu, s, B_std[rank_a], est['ff3']['alpha'], est['ff3']['betas'], H,
                           alternative_B_std={k: B_std[k] for k in ranks})
    scored = oc['scored']
    xf = np.where(future_valid, future_log, 0.)
    z = (xf - mu[None, :]) / s[None, :]
    S = est['signs']
    extra = {}
    for K in ranks:
        e_std, _ = broad_residuals(z, future_valid, B_std[K])
        e_ret = e_std * s[None, :]
        extra[str(K)] = dict(random_signed=rr.random_signed_stats(e_ret[:, scored], S[scored], None) if scored.sum() > 2 else None,
                             next_direction=rr.portfolio_variance_ratio(e_std, est['next_directions'][K], scored))
    excess = xf - np.log1p(rf_future)[:, None]
    e_fixed = excess - est['ff3']['alpha'][None, :] - ff_future @ est['ff3']['betas'].T
    extra['ff3'] = dict(random_signed=rr.random_signed_stats(e_fixed[:, scored], S[scored], None) if scored.sum() > 2 else None,
                        next_direction=rr.portfolio_variance_ratio(e_fixed / s[None, :], est['next_directions']['ff3'], scored))
    return oc, extra


def primary_record_check(date, permno, est, oc):
    """Compare with the application record of the same date, if present."""
    js = PRIMARY_RECORDS / f'{date}.json'
    if not js.is_file():
        return dict(available=False)
    p = json.loads(js.read_text())
    z = np.load(PRIMARY_RECORDS / f'{date}.npz')
    out = dict(available=True, same_issuers=bool(len(z['permno']) == len(permno) and np.array_equal(z['permno'], permno)),
               icp2_equal=bool(p['rank_diagnostics']['icp2_rank'] == est['rules']['icp2_rank']), sign_equal=bool(p['rank_diagnostics']['sign_rank'] == est['rules']['sign_rank']),
               primary_rank=p['rank'], primary_insample_equal_weight={k: v['equal_weight_residual_variance_ratio'] for k, v in p['rank_diagnostics']['alternative_rank_content'].items()})
    out['insample_equal_weight_max_abs_diff'] = max(abs(est['curves']['insample_equal_weight'][int(k)] - v) for k, v in out['primary_insample_equal_weight'].items()
                                                    if int(k) < len(est['curves']['insample_equal_weight']))
    if oc is not None and out['same_issuers'] and 'rv_ff3' in z.files:
        a, b = oc['rv_ff3'], z['rv_ff3']; ok = np.isfinite(a) & np.isfinite(b)
        out['rv_ff3_max_abs_rel_diff'] = float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(b[ok], 1e-300))) if ok.any() else None
        out['scored_equal'] = bool(np.array_equal(oc['scored'], z['scored']))
        for k, v in oc['alternative_rank_diversifiability'].items():
            pv = (p.get('outcomes') or {}).get('alternative_rank_diversifiability', {}).get(k)
            if pv is not None:
                out.setdefault('equal_weight_diversifiability_abs_diff', {})[k] = abs(v - pv)
    return out


def run_date(cache, t_idx, config=CONFIG, out=RECORDS, roles=None, score=True, stem_suffix=''):
    started = time.perf_counter(); stamp = datetime.now(timezone.utc).isoformat()
    T, H = config['T'], config['H']
    date = str(cache.dates[t_idx])
    idx, uinfo = eligible_universe(cache, t_idx, T)
    rows = slice(t_idx - T + 1, t_idx + 1)
    x = np.asarray(cache.ret[rows][:, idx], dtype=np.float64)
    ffw = np.asarray(cache.ff[rows], dtype=np.float64)
    base_seed = int(np.datetime64(cache.dates[t_idx], 'D').astype(np.int64))
    est = estimate_rank_rule_panel(x, ffw[:, :3], ffw[:, 3], base_seed, t_idx, config, roles)
    permno = cache.permnos[idx]
    record = dict(est['forecasts'])
    record.update(permno=permno, permco=np.asarray(cache.permco[t_idx])[idx], mktcap=np.asarray(cache.mktcap[t_idx], dtype=np.float64)[idx])
    summary = dict(version=VERSION, date=date, t_idx=int(t_idx), T=T, N=int(x.shape[1]), universe=uinfo, rules=est['rules'], curves=est['curves'], models=est['models'],
                   ff3_insample=est['ff3']['insample'], diagnostic_ranks=est['diagnostic_ranks'], base_seed=base_seed, started_utc=stamp, config=dict(config),
                   source_hashes=source_hashes())
    outcome_available = bool(t_idx + H < len(cache)) and score
    summary['outcome_available'] = outcome_available
    oc = None
    if outcome_available:
        frows = slice(t_idx + 1, t_idx + 1 + H)
        future = np.asarray(cache.ret[frows][:, idx], dtype=np.float64)
        fvalid = np.asarray(cache.valid[frows][:, idx])
        fff = np.asarray(cache.ff[frows], dtype=np.float64)
        oc, extra = outcome_diagnostics(future, fvalid, fff[:, :3], fff[:, 3], est, H)
        record.update(rv_ff3=oc['rv_ff3'], rv_ff3_fixed=oc['rv_ff3_fixed'], rv_total=oc['rv_total'], scored=oc['scored'])
        for k, v in oc['alternative_rank_rv'].items():
            record[f'rv_broad_k{k}'] = v
        summary['outcomes'] = dict(scored_count=oc['scored_count'], outcome_first=str(cache.dates[t_idx + 1]), outcome_last=str(cache.dates[t_idx + H]),
                                   equal_weight=dict({k: v for k, v in oc['alternative_rank_diversifiability'].items()},
                                                     ff3=oc['diversifiability_ff3_fixed'], total=oc['diversifiability_total']),
                                   random_signed={k: v['random_signed'] for k, v in extra.items()},
                                   next_direction={k: v['next_direction'] for k, v in extra.items()})
    summary['primary_record_check'] = primary_record_check(date, permno, est, oc)
    summary['timings'] = est['timings']
    summary['seconds'] = round(time.perf_counter() - started, 2)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / (date + stem_suffix)
    np.savez_compressed(str(stem) + '.npz', **{k: np.asarray(v) for k, v in record.items()})
    tmp = Path(str(stem) + '.json.tmp')
    tmp.write_text(json.dumps(summary, indent=1, default=_plain))
    tmp.replace(str(stem) + '.json')
    return summary


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (set, tuple)):
        return list(v)
    raise TypeError(str(type(v)))


def safe_run_date(cache, t_idx, config=CONFIG, out=RECORDS):
    try:
        return run_date(cache, t_idx, config, out)
    except Exception as exc:
        date = str(cache.dates[t_idx])
        failure = dict(date=date, t_idx=int(t_idx), status='failed_retained', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                       utc=datetime.now(timezone.utc).isoformat())
        out.mkdir(parents=True, exist_ok=True)
        (out / (date + '.FAILED.json')).write_text(json.dumps(failure, indent=1))
        return failure
