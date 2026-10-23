"""Restricted PCA factor-GARCH benchmark with an approximate payoff mapping.

Established ingredients: correlation PCA, variance-targeted Gaussian GARCH QML,
and expected-variance forecasts. This is not DCC, factor SV, or an exact GARCH
compounded-payoff law. Replacing future random variances by their expectations
before applying Gaussian exponential moments is an explicit approximation.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        return lambda fn: fn

VERSION = 'correlation_pca_targeted_garch_plugin_20261015_v1'
MAXIMUM_RANK = 12
MAX_PERSISTENCE = .995
STARTS = ((.95, .05/.95), (.85, .15/.85), (.25, .05/.25))


@njit(cache=True)
def _objective_ab(alpha, beta, squares):
    """Average Gaussian QML without constants; normalized variance target = 1."""
    h = 1.0
    da = 0.0
    db = 0.0
    objective = 0.0
    gradient_a = 0.0
    gradient_b = 0.0
    for index in range(len(squares)):
        if index:
            previous_h = h
            h = 1.0-alpha-beta + alpha*squares[index-1] + beta*previous_h
            da = -1.0+squares[index-1]+beta*da
            db = -1.0+previous_h+beta*db
        objective += .5*(np.log(h)+squares[index]/h)
        multiplier = .5*(1.0-squares[index]/h)/h
        gradient_a += multiplier*da
        gradient_b += multiplier*db
    n = len(squares)
    return objective/n, gradient_a/n, gradient_b/n, h


def _objective(parameters, squares):
    rho, share = parameters
    alpha, beta = rho*share, rho*(1.-share)
    objective, ga, gb, _ = _objective_ab(alpha, beta, squares)
    return objective, np.array([share*ga+(1-share)*gb, rho*(ga-gb)])


def fit_series(values):
    """Variance target and three fixed QML starts; retain best finite candidate.

    The static alpha=beta=0 candidate is always included. A best finite
    nonconverged optimization iterate is retained and disclosed, not silently
    replaced or excluded from later evaluation.
    """
    y = np.ascontiguousarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) < 12 or not np.isfinite(y).all():
        raise ValueError('A complete finite one-dimensional history is required')
    target = float(np.mean(y*y))
    if target == 0.:
        return {'alpha': 0., 'beta': 0., 'target': 0., 'next_variance': 0.,
                'objective': .5, 'converged': True, 'static_selected': True,
                'optimizer_failures': 0, 'iterations': 0}
    squares = np.ascontiguousarray(y*y/target)
    best = {'alpha': 0., 'beta': 0., 'target': target, 'next_variance': target,
            'objective': float(_objective_ab(0., 0., squares)[0]),
            'converged': True, 'static_selected': True, 'iterations': 0}
    failures = 0
    for start in STARTS:
        result = minimize(_objective, np.array(start), args=(squares,), jac=True,
                          method='L-BFGS-B', bounds=((0., MAX_PERSISTENCE), (0., 1.)),
                          options={'maxiter': 150, 'ftol': 1e-11, 'gtol': 1e-7, 'maxls': 40})
        failures += int(not result.success)
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            continue
        rho, share = result.x
        alpha, beta = float(rho*share), float(rho*(1.-share))
        value, _, _, last_h = _objective_ab(alpha, beta, squares)
        if value < best['objective']:
            h_next = (1.-rho)+alpha*squares[-1]+beta*last_h
            best = {'alpha': alpha, 'beta': beta, 'target': target,
                    'next_variance': float(target*h_next), 'objective': float(value),
                    'converged': bool(result.success), 'static_selected': False,
                    'iterations': int(result.nit)}
    best['optimizer_failures'] = failures
    return best


def expected_variance_sum(model, horizon):
    """Sum E_t h_(t+j), with next_variance already using the last observed shock."""
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    rho = model['alpha']+model['beta']
    h = model['next_variance']
    answer = 0.
    for _ in range(horizon):
        answer += h
        h = model['target']+rho*(h-model['target'])
    return float(answer)


def correlation_pca(history, maximum=MAXIMUM_RANK):
    x = np.asarray(history, dtype=float)
    if x.ndim != 2 or min(x.shape) < 12 or not np.isfinite(x).all():
        raise ValueError('Need a complete finite log-return history with both dimensions >= 12')
    t, n = x.shape
    mu = x.mean(axis=0)
    centered = x-mu
    scale = np.sqrt(np.mean(centered*centered, axis=0))
    if np.any(scale*scale <= 1e-14):
        raise ValueError('Degenerate return coordinate')
    standardized = centered/scale
    covariance = standardized.T@standardized/t
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    maximum = min(maximum, n-1, t-3)
    tail = np.maximum((eigenvalues.sum()-np.r_[0., np.cumsum(eigenvalues[:maximum])])/n, 1e-16)
    penalty = (n+t)/(n*t)*np.log(min(n, t))
    criterion = np.log(tail)+np.arange(maximum+1)*penalty
    rank = int(np.argmin(criterion))
    scores = standardized@eigenvectors[:, :rank]
    loadings = scale[:, None]*eigenvectors[:, :rank]
    residuals = centered-scores@loadings.T
    # Estimated PCA factors are not fixed exogenous regressors: heuristic only.
    df_multiplier = t/(t-rank-1)
    residuals = residuals*np.sqrt(df_multiplier)
    return mu, loadings, scores, residuals, {'rank': rank, 'rank_maximum': maximum,
                                             'residual_df_multiplier': df_multiplier}


def gaussian_payoff_plugin(mean_sum, common_variance_sum, residual_variance_sum):
    """Exact Gaussian formula for a deterministic-variance surrogate law only."""
    m, c, d = np.broadcast_arrays(mean_sum, common_variance_sum, residual_variance_sum)
    with np.errstate(over='raise', invalid='raise'):
        q = np.exp(2*m+2*c+d)*np.expm1(d)
        total = np.exp(2*m+c+d)*np.expm1(c+d)
        mean_return = np.expm1(m+.5*(c+d))
        second = np.exp(2*m+2*(c+d))
    forecast = {'q': q, 'total_variance': total, 'mean_return': mean_return,
                'gross_second_moment': second}
    if not all(np.isfinite(v).all() for v in forecast.values()):
        raise ArithmeticError('Nonfinite Gaussian plug-in payoff moment')
    return forecast


def estimate_garch(history, horizon=21):
    """Training-only restricted factor-GARCH plug-in; returns forecast, diagnostics."""
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    started = time.perf_counter()
    mu, loadings, scores, residuals, pca = correlation_pca(history)
    factor_fits = [fit_series(scores[:, j]) for j in range(scores.shape[1])]
    residual_fits = [fit_series(residuals[:, j]) for j in range(residuals.shape[1])]
    factor_sums = np.array([expected_variance_sum(fit, horizon) for fit in factor_fits])
    common_sum = (loadings*loadings)@factor_sums
    residual_sum = np.array([expected_variance_sum(fit, horizon) for fit in residual_fits])
    forecast = gaussian_payoff_plugin(horizon*mu, common_sum, residual_sum)
    fits = factor_fits+residual_fits
    persistence = np.array([fit['alpha']+fit['beta'] for fit in fits])
    diagnostics = {
        'garch_rank': pca['rank'], 'garch_rank_maximum': pca['rank_maximum'],
        'garch_rank_at_cap': bool(pca['rank'] == pca['rank_maximum']),
        'garch_residual_df_multiplier': pca['residual_df_multiplier'],
        'garch_series': len(fits),
        'garch_selected_converged': int(sum(fit['converged'] for fit in fits)),
        'garch_failed_optimizer_starts': int(sum(fit['optimizer_failures'] for fit in fits)),
        'garch_static_series': int(sum(fit['static_selected'] for fit in fits)),
        'garch_persistence_at_cap': int(np.sum(persistence >= MAX_PERSISTENCE-1e-7)),
        'garch_max_persistence': float(persistence.max()),
        'garch_mean_alpha': float(np.mean([fit['alpha'] for fit in fits])),
        'garch_mean_beta': float(np.mean([fit['beta'] for fit in fits])),
        'garch_total_selected_iterations': int(sum(fit['iterations'] for fit in fits)),
        'garch_numba': NUMBA_AVAILABLE,
        'garch_seconds': float(time.perf_counter()-started),
        'garch_horizon_mapping': 'deterministic_expected_variance_gaussian_plugin',
        'garch_conditional_covariance': 'independent_PCA_factor_and_diagonal_residual_GARCH',
        'garch_df_rule': 'T/(T-k-1); estimated-factor heuristic, not unbiasedness claim',
    }
    return forecast, diagnostics
