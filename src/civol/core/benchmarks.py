"""FF3 common space and the comparator factorial.

FF3 regressions use the issuer's daily log return in excess of log(1+RF) on the
three Fama-French factors with an intercept. The literature-exact variant on
simple excess returns is recorded separately for the standard one-month measure.
"""
from __future__ import annotations
import numpy as np


def ols(y, X):
    """Vectorized OLS of each column of y (T x N) on X (T x k) with intercept."""
    T = len(y)
    Xc = np.column_stack([np.ones(T), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    return beta, resid


def ff3_space(x_log, ff, rf):
    """Formation-window FF3 decomposition. Returns intercepts, betas (N x 3),
    residuals (T x N, df-corrected) and the excess-return panel used."""
    T, N = x_log.shape
    y = x_log - np.log1p(rf)[:, None]
    beta, resid = ols(y, ff)
    df = T / (T - 4)
    resid = resid * np.sqrt(df)
    return dict(alpha=beta[0], betas=beta[1:].T, residuals=resid, excess=y, residual_df_multiplier=df)


def ff3_realized(x_log, ff, rf, window, horizon=21, simple=None):
    """Standard measure: contemporaneous FF3 regression over the last `window`
    sessions; residual variance with window-4 degrees of freedom, times horizon."""
    y = (x_log - np.log1p(rf)[:, None])[-window:]
    _, resid = ols(y, ff[-window:])
    var_log = np.sum(resid * resid, axis=0) / (window - 4)
    out = dict(forecast=horizon * var_log)
    if simple is not None:
        ys = (simple - rf[:, None])[-window:]
        _, rs = ols(ys, ff[-window:])
        out['forecast_simple'] = horizon * np.sum(rs * rs, axis=0) / (window - 4)
    return out


def total_variance_1m(z, horizon=21, window=21):
    zz = z[-window:]
    return horizon * np.sum((zz - zz.mean(0)) ** 2, axis=0) / (window - 1)
