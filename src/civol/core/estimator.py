"""End-to-end conditional IVOL estimation for one complete log-return panel, and
the formation-date wrapper that reads the CRSP cache, scores outcomes and
writes an immutable record.

estimate_panel(x, ff, rf, seed) is the single code path used by both the
historical application and the truth-known simulation.
"""
from __future__ import annotations
import hashlib, json, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from civol.core.paths import HERE, RECORDS
from civol.core.common_space import common_space, residual_common_content
from civol.core.private_dynamics import fit_private_dynamics, factor_garch, garch_per_issuer, ewma_forecast, realized_1m
from civol.core.horizon import paired_moments, surrogate_moments
from civol.core.benchmarks import ff3_space, ff3_realized, total_variance_1m
from civol.core.outcomes import realized_outcomes
from civol.core.universe import eligible_universe

VERSION = 'thesis_conditional_ivol_v1'
CONFIG = dict(T=750, H=21, pairs=16384, batches=8, kmax=20, sign_draws=64, rank_rule='icp2', alternative_ranks=(10, 20), variant_rank=10,
              transitory_driver='residual', level_driver='total', factor_bound_multiple=25., ewma_decay=.94, ff3_short=21, ff3_long=250, backend='numpy')
SOURCE_FILES = ['common_space.py', 'private_dynamics.py', 'horizon.py', 'horizon_gpu.py', 'benchmarks.py', 'outcomes.py', 'universe.py', 'estimator.py', 'crsp_cache.py', 'adaptive_garch.py', 'paths.py']


def source_hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _timer(timings, name, started):
    timings[name] = round(time.perf_counter() - started, 3)
    return time.perf_counter()


def _empty_factor_garch():
    z = np.zeros(0)
    return dict(omega=z, alpha=z, beta=z, g_next=z, target=z, converged=0, optimizer_failures=0, static_selected=0)


def estimate_panel(x, ff, rf, seed, config=CONFIG, simple=None):
    """x: T x N complete log returns; ff: T x 3 factor returns; rf: T risk-free.
    Returns per-issuer forecasts for cIVOL, FF3-paired and the comparator
    factorial, plus model diagnostics. All inputs end at the formation date."""
    x = np.asarray(x, float); ff = np.asarray(ff, float); rf = np.asarray(rf, float)
    T, N = x.shape; H = config['H']; backend = config.get('backend', 'numpy')
    drivers = (config.get('transitory_driver', 'total'), config.get('level_driver', 'residual'))
    dyn = dict(transitory=drivers[0], level=drivers[1])
    timings = {}; t0 = time.perf_counter()
    # --- broad common space
    cs = common_space(x, kmax=config['kmax'], sign_draws=config['sign_draws'], rule=config.get('rank_rule', 'icp2'),
                      alternative_ranks=tuple(config.get('alternative_ranks', (10, 20))), variant_rank=config.get('variant_rank'))
    t0 = _timer(timings, 'common_space', t0)
    mu, s, b, e = cs['mu'], cs['s'], cs['loadings'], cs['residuals']
    z = x - mu
    fg = factor_garch(cs['F']) if cs['rank'] else _empty_factor_garch()
    t0 = _timer(timings, 'factor_garch', t0)
    reference = np.sum(b * b, axis=1) + cs['d0']
    pdyn = fit_private_dynamics(z, e, cs['d0'], reference, component=True, **dyn)
    pdyn1 = fit_private_dynamics(z, e, cs['d0'], reference, component=False, **dyn)      # nested one-component law, same driver
    pdynT = fit_private_dynamics(z, e, cs['d0'], reference, component=False, transitory='total', level='total')   # accepted research law
    t0 = _timer(timings, 'private_dynamics', t0)
    bound_multiple = float(config.get('factor_bound_multiple', 25.))
    model = dict(mu=mu, loadings=b, d=pdyn['d'], reference=reference, h_next=pdyn['h_next'], m_next=pdyn['m_next'], coefficients=pdyn['coefficients'],
                 drivers=drivers, factor_omega=fg['omega'], factor_alpha=fg['alpha'], factor_beta=fg['beta'], g_next=fg['g_next'], factor_target=fg['target'],
                 factor_bound_multiple=bound_multiple)
    pm = paired_moments(model, H, config['pairs'], config['batches'], seed, backend=backend)
    sm = surrogate_moments(model, H)
    model1 = dict(model, d=pdyn1['d'], h_next=pdyn1['h_next'], m_next=pdyn1['m_next'], coefficients=pdyn1['coefficients'])
    pm1 = paired_moments(model1, H, config['pairs'], config['batches'], seed + 3, backend=backend)
    modelT = dict(model, d=pdynT['d'], h_next=pdynT['h_next'], m_next=pdynT['m_next'], coefficients=pdynT['coefficients'], drivers=('total', 'total'))
    pmT = paired_moments(modelT, H, config['pairs'], config['batches'], seed + 4, backend=backend)
    t0 = _timer(timings, 'paired_integration', t0)
    # --- Fixed-rank comparison (same estimator at K = variant_rank)
    variant_forecasts = {}; variant_summary = None
    if cs.get('variant') is not None:
        vr = cs['variant']
        fg_v = factor_garch(vr['F']) if vr['rank'] else _empty_factor_garch()
        ref_v = np.sum(vr['loadings'] ** 2, axis=1) + vr['d0']
        pd_v = fit_private_dynamics(z, vr['residuals'], vr['d0'], ref_v, component=True, **dyn)
        model_v = dict(mu=mu, loadings=vr['loadings'], d=pd_v['d'], reference=ref_v, h_next=pd_v['h_next'], m_next=pd_v['m_next'], coefficients=pd_v['coefficients'],
                       drivers=drivers, factor_omega=fg_v['omega'], factor_alpha=fg_v['alpha'], factor_beta=fg_v['beta'], g_next=fg_v['g_next'], factor_target=fg_v['target'],
                       factor_bound_multiple=bound_multiple)
        pm_v = paired_moments(model_v, H, config['pairs'], config['batches'], seed + 2, backend=backend)
        sm_v = surrogate_moments(model_v, H)
        variant_forecasts = dict(civolk10_q=pm_v['q'], civolk10_ipv=pm_v['ipv'], civolk10_total_variance=pm_v['total_variance'],
                                 civolk10_q_mcse=pm_v['q_mcse'], civolk10_ipv_mcse=pm_v['ipv_mcse'], civolk10_surrogate_ipv=sm_v['ipv'],
                                 pcak10_1m=realized_1m(vr['residuals'], H), pcak10_ewma=ewma_forecast(vr['residuals'], H, config['ewma_decay']))
        variant_summary = dict(rank=vr['rank'], pooled=dict(coefficients=pd_v['coefficients'].tolist(), objective=pd_v['objective'], converged=pd_v['converged'],
                                                             static_selected=pd_v['static_selected']),
                               factor_garch=dict(alpha=fg_v['alpha'].tolist(), beta=fg_v['beta'].tolist(), converged=fg_v['converged']),
                               integration=dict(backend=pm_v.get('backend'), median_relative_q_mcse=float(np.median(pm_v['q_mcse'] / pm_v['q'])),
                                                median_compounding_factor=float(np.median(pm_v['q'] / pm_v['ipv']))))
        t0 = _timer(timings, 'variant_k10', t0)
    # --- FF3 common space and FF3-paired
    ffs = ff3_space(x, ff, rf)
    fg_ff = factor_garch(ff)
    d0_ff = np.mean(ffs['residuals'] ** 2, axis=0)
    reference_ff = (ffs['betas'] ** 2) @ fg_ff['target'] + d0_ff
    pdyn_ff = fit_private_dynamics(z, ffs['residuals'], d0_ff, reference_ff, component=True, **dyn)
    model_ff = dict(mu=mu, loadings=ffs['betas'], d=pdyn_ff['d'], reference=reference_ff, h_next=pdyn_ff['h_next'], m_next=pdyn_ff['m_next'], coefficients=pdyn_ff['coefficients'],
                    drivers=drivers, factor_omega=fg_ff['omega'], factor_alpha=fg_ff['alpha'], factor_beta=fg_ff['beta'], g_next=fg_ff['g_next'], factor_target=fg_ff['target'],
                    factor_bound_multiple=bound_multiple)
    pm_ff = paired_moments(model_ff, H, config['pairs'], config['batches'], seed + 1, backend=backend)
    sm_ff = surrogate_moments(model_ff, H)
    t0 = _timer(timings, 'ff3_paired', t0)
    # --- comparator factorial
    ff3_1m = ff3_realized(x, ff, rf, config['ff3_short'], H, simple=simple)
    ff3_12m = ff3_realized(x, ff, rf, config['ff3_long'], H)
    ff3_garch = garch_per_issuer(ffs['residuals'], H)
    pca_garch = garch_per_issuer(e, H)
    forecasts = dict(
        civol_q=pm['q'], civol_ipv=pm['ipv'], civol_total_variance=pm['total_variance'], civol_mean_return=pm['mean_return'],
        civol_q_mcse=pm['q_mcse'], civol_ipv_mcse=pm['ipv_mcse'], civol_total_variance_mcse=pm['total_variance_mcse'],
        civol_surrogate_q=sm['q'], civol_surrogate_ipv=sm['ipv'], civol_common_variance_sum=sm['common_variance_sum'],
        civol1c_q=pm1['q'], civol1c_ipv=pm1['ipv'], civol1c_q_mcse=pm1['q_mcse'], civol1c_ipv_mcse=pm1['ipv_mcse'],
        civol1cT_q=pmT['q'], civol1cT_ipv=pmT['ipv'], civol1cT_q_mcse=pmT['q_mcse'], civol1cT_ipv_mcse=pmT['ipv_mcse'],
        ff3paired_q=pm_ff['q'], ff3paired_ipv=pm_ff['ipv'], ff3paired_total_variance=pm_ff['total_variance'], ff3paired_q_mcse=pm_ff['q_mcse'],
        ff3paired_ipv_mcse=pm_ff['ipv_mcse'], ff3paired_surrogate_q=sm_ff['q'], ff3paired_surrogate_ipv=sm_ff['ipv'],
        ff3_1m=ff3_1m['forecast'], ff3_12m=ff3_12m['forecast'], ff3_ewma=ewma_forecast(ffs['residuals'], H, config['ewma_decay']),
        ff3_garch=ff3_garch['forecast'], pca_1m=realized_1m(e, H), pca_ewma=ewma_forecast(e, H, config['ewma_decay']), pca_garch=pca_garch['forecast'],
        total_1m=total_variance_1m(z, H))
    if 'forecast_simple' in ff3_1m:
        forecasts['ff3_1m_simple'] = ff3_1m['forecast_simple']
    forecasts.update(variant_forecasts)
    t0 = _timer(timings, 'comparators', t0)
    content = dict(pca=residual_common_content(e), ff3=residual_common_content(ffs['residuals']))
    t0 = _timer(timings, 'residual_content', t0)
    issuer = dict(mu=mu, s=s, d=pdyn['d'], h_next=pdyn['h_next'], m_next=pdyn['m_next'], loadings=b, d_ff3=pdyn_ff['d'], h_next_ff3=pdyn_ff['h_next'],
                  d_1c=pdyn1['d'], h_next_1c=pdyn1['h_next'],
                  ff3_alpha=ffs['alpha'], ff3_betas=ffs['betas'], pca_garch_alpha=pca_garch['alpha'], pca_garch_beta=pca_garch['beta'],
                  ff3_garch_alpha=ff3_garch['alpha'], ff3_garch_beta=ff3_garch['beta'])
    integration = {k: pm.get(k) for k in ('pairs', 'batches', 'seed', 'lower_contacts', 'upper_contacts', 'backend', 'device', 'gpu_error',
                                           'common_upper_contacts', 'g_next_clipped_factors', 'factor_bound_multiple')}
    integration['ff3paired_common_upper_contacts'] = pm_ff.get('common_upper_contacts'); integration['ff3paired_g_next_clipped_factors'] = pm_ff.get('g_next_clipped_factors')
    integration.update(median_relative_q_mcse=float(np.median(pm['q_mcse'] / pm['q'])), max_relative_q_mcse=float(np.max(pm['q_mcse'] / pm['q'])),
                       median_relative_ipv_mcse=float(np.median(pm['ipv_mcse'] / pm['ipv'])), max_relative_ipv_mcse=float(np.max(pm['ipv_mcse'] / pm['ipv'])),
                       median_compounding_factor=float(np.median(pm['q'] / pm['ipv'])), median_surrogate_ratio=float(np.median(pm['q'] / sm['q'])),
                       ff3paired_backend=pm_ff.get('backend'), ff3paired_gpu_error=pm_ff.get('gpu_error'))
    summary = dict(version=VERSION, T=T, N=N, rank=cs['rank'], rank_diagnostics=cs['diagnostics'],
                   pooled=dict(coefficients=pdyn['coefficients'].tolist(), objective=pdyn['objective'], converged=pdyn['converged'], static_selected=pdyn['static_selected'],
                               history=pdyn['history'], lower_contacts=pdyn['lower_contacts'], upper_contacts=pdyn['upper_contacts'],
                               median_m_next=float(np.median(pdyn['m_next'])), median_h_next=float(np.median(pdyn['h_next'])),
                               transitory_driver=drivers[0], level_driver=drivers[1]),
                   pooled_one_component=dict(coefficients=pdyn1['coefficients'].tolist(), objective=pdyn1['objective'], converged=pdyn1['converged'],
                                             static_selected=pdyn1['static_selected'], median_h_next=float(np.median(pdyn1['h_next'])),
                                             objective_gain_component=float(pdyn1['objective'] - pdyn['objective']), transitory_driver=drivers[0]),
                   pooled_accepted_law=dict(coefficients=pdynT['coefficients'].tolist(), objective=pdynT['objective'], converged=pdynT['converged'],
                                            static_selected=pdynT['static_selected'], median_h_next=float(np.median(pdynT['h_next'])), transitory_driver='total'),
                   pooled_ff3=dict(coefficients=pdyn_ff['coefficients'].tolist(), objective=pdyn_ff['objective'], converged=pdyn_ff['converged'], static_selected=pdyn_ff['static_selected'],
                                   history=pdyn_ff['history']),
                   factor_garch=dict(alpha=fg['alpha'].tolist(), beta=fg['beta'].tolist(), target=fg['target'].tolist(), g_next=fg['g_next'].tolist(),
                                     converged=fg['converged'], optimizer_failures=fg['optimizer_failures'], static_selected=fg['static_selected']),
                   factor_garch_ff3=dict(alpha=fg_ff['alpha'].tolist(), beta=fg_ff['beta'].tolist(), target=fg_ff['target'].tolist(), g_next=fg_ff['g_next'].tolist(),
                                         converged=fg_ff['converged'], optimizer_failures=fg_ff['optimizer_failures']),
                   integration=integration, residual_content=content, variant=variant_summary, timings=timings, seed=int(seed))
    return dict(forecasts=forecasts, issuer=issuer, summary=summary, B_std=cs['B'], alternative_B_std=cs['alternative_B_std'], model=model, model_ff3=model_ff)


def run_date(cache, t_idx, config=CONFIG, pilot=False, out=RECORDS):
    """Estimate every eligible issuer at session t_idx and score the next H sessions."""
    started = time.perf_counter(); stamp = datetime.now(timezone.utc).isoformat()
    T, H = config['T'], config['H']
    date = str(cache.dates[t_idx])
    idx, uinfo = eligible_universe(cache, t_idx, T)
    rows = slice(t_idx - T + 1, t_idx + 1)
    x = np.asarray(cache.ret[rows][:, idx], dtype=np.float64)
    ffw = np.asarray(cache.ff[rows], dtype=np.float64)
    seed = int(np.datetime64(cache.dates[t_idx], 'D').astype(np.int64))
    est = estimate_panel(x, ffw[:, :3], ffw[:, 3], seed, config, simple=np.expm1(x))
    record = dict(est['forecasts'])
    record.update({f'issuer_{k}': v for k, v in est['issuer'].items()})
    record.update(permno=cache.permnos[idx], permco=np.asarray(cache.permco[t_idx])[idx], mktcap=np.asarray(cache.mktcap[t_idx], dtype=np.float64)[idx])
    summary = est['summary']
    summary.update(date=date, t_idx=int(t_idx), universe=uinfo, pilot=bool(pilot), started_utc=stamp, config=dict(config), source_hashes=source_hashes())
    outcome_available = t_idx + H < len(cache)
    summary['outcome_available'] = bool(outcome_available)
    if outcome_available:
        frows = slice(t_idx + 1, t_idx + 1 + H)
        future = np.asarray(cache.ret[frows][:, idx], dtype=np.float64)
        fvalid = np.asarray(cache.valid[frows][:, idx])
        fff = np.asarray(cache.ff[frows], dtype=np.float64)
        oc = realized_outcomes(future, fvalid, fff[:, :3], fff[:, 3], est['issuer']['mu'], est['issuer']['s'], est['B_std'],
                               est['issuer']['ff3_alpha'], est['issuer']['ff3_betas'], H, alternative_B_std=est['alternative_B_std'])
        for k in ('rv_ff3', 'rv_ff3_fixed', 'rv_broad', 'rv_total', 'scored'):
            record[k] = oc[k]
        record['realized_factors'] = oc['realized_factors']
        for k, v in oc['alternative_rank_rv'].items():
            record[f'rv_broad_k{k}'] = v
        summary['outcomes'] = dict(scored_count=oc['scored_count'], outcome_first=str(cache.dates[t_idx + 1]), outcome_last=str(cache.dates[t_idx + H]),
                                   diversifiability_ff3_fixed=oc['diversifiability_ff3_fixed'], diversifiability_broad=oc['diversifiability_broad'],
                                   diversifiability_total=oc['diversifiability_total'], portfolio_rv_ff3_fixed=oc['portfolio_rv_ff3_fixed'],
                                   portfolio_rv_broad=oc['portfolio_rv_broad'], alternative_rank_diversifiability=oc['alternative_rank_diversifiability'])
    summary['seconds'] = round(time.perf_counter() - started, 2)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / date
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


def safe_run_date(cache, t_idx, config=CONFIG, pilot=False, out=RECORDS):
    try:
        return run_date(cache, t_idx, config, pilot, out)
    except Exception as exc:
        date = str(cache.dates[t_idx])
        failure = dict(date=date, t_idx=int(t_idx), status='failed_retained', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                       utc=datetime.now(timezone.utc).isoformat())
        out.mkdir(parents=True, exist_ok=True)
        (out / (date + '.FAILED.json')).write_text(json.dumps(failure, indent=1))
        return failure
