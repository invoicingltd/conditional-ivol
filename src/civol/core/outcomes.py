"""Realized outcome proxies and the diversifiability diagnostic for the 21
sessions after a formation date. Nothing here is used by any forecast.
"""
from __future__ import annotations
import numpy as np
from civol.core.benchmarks import ols


def broad_residuals(z_future, valid, B_std):
    """Residuals under formation loadings with realized factors extracted each day
    by cross-sectional least squares on the available issuers. z_future is H x N
    standardized (by formation s) demeaned returns with zeros where invalid."""
    H, N = z_future.shape
    K = B_std.shape[1]
    f_hat = np.zeros((H, K)); e = np.zeros((H, N))
    for u in range(H):
        avail = valid[u]
        if K and avail.sum() > K:
            f, *_ = np.linalg.lstsq(B_std[avail], z_future[u, avail], rcond=None)
            f_hat[u] = f
        e[u] = z_future[u] - B_std @ f_hat[u]
    return e, f_hat


def portfolio_ratio(e, rv, scored):
    """N * RV(equal-weight residual portfolio) / mean_i RV_i over scored issuers."""
    n = int(scored.sum())
    if n < 2:
        return np.nan, np.nan
    p = e[:, scored].mean(1)
    rv_p = float(np.sum(p * p))
    return float(n * rv_p / np.nanmean(rv[scored])), rv_p


def realized_outcomes(future_log, future_valid, ff_future, rf_future, mu, s, B_std, ff_alpha, ff_betas, horizon=21, alternative_B_std=None):
    """future_log: H x N log returns (NaN where invalid); future_valid: H x N bool.

    Returns per-issuer realized variances under each residual definition, the
    scored mask (all H sessions valid), and portfolio diversifiability ratios.
    """
    H, N = future_log.shape
    scored = future_valid.all(0)
    out = dict(scored=scored, scored_count=int(scored.sum()))
    x = np.where(future_valid, future_log, 0.)
    excess = x - np.log1p(rf_future)[:, None]
    # RV-FF3: contemporaneous regression on the H sessions (standard realized IVOL)
    rv_ff3 = np.full(N, np.nan)
    if scored.sum() > 4:
        _, resid = ols(excess[:, scored], ff_future)
        rv_ff3[scored] = horizon * np.sum(resid * resid, axis=0) / (H - 4)
    out['rv_ff3'] = rv_ff3
    # RV-FF3-fixed: formation alpha/betas
    e_fixed = excess - ff_alpha[None, :] - ff_future @ ff_betas.T
    rv_ff3_fixed = np.where(scored, np.sum(e_fixed * e_fixed, axis=0), np.nan)
    out['rv_ff3_fixed'] = rv_ff3_fixed
    # RV-broad: realized factors by cross-sectional least squares on formation loadings
    z = (x - mu[None, :]) / s[None, :]
    e_broad, f_hat = broad_residuals(z, future_valid, B_std)
    e_broad = e_broad * s[None, :]
    rv_broad = np.where(scored, np.sum(e_broad * e_broad, axis=0), np.nan)
    out['rv_broad'] = rv_broad
    out['realized_factors'] = f_hat
    # realized total variance (no common space removed), centered at the formation mean
    xt = x - mu[None, :]
    out['rv_total'] = np.where(scored, np.sum(xt * xt, axis=0), np.nan)
    out['diversifiability_ff3_fixed'], out['portfolio_rv_ff3_fixed'] = portfolio_ratio(e_fixed, rv_ff3_fixed, scored)
    out['diversifiability_broad'], out['portfolio_rv_broad'] = portfolio_ratio(e_broad, rv_broad, scored)
    out['diversifiability_total'], _ = portfolio_ratio(xt, out['rv_total'], scored)
    # K-sensitivity of the broad residual definition
    out['alternative_rank_diversifiability'] = {}
    out['alternative_rank_rv'] = {}
    for k, Bk in (alternative_B_std or {}).items():
        ek, _ = broad_residuals(z, future_valid, Bk)
        ek = ek * s[None, :]
        rvk = np.where(scored, np.sum(ek * ek, axis=0), np.nan)
        out['alternative_rank_diversifiability'][str(k)], _ = portfolio_ratio(ek, rvk, scored)
        out['alternative_rank_rv'][str(k)] = rvk
    return out
