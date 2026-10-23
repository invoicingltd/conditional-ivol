"""Private variance dynamics: pooled bounded own-return feedback with a slowly
adapting level component, per-issuer GARCH, EWMA, and factor GARCH.

Two-component bounded recursion (per issuer), with transitory driver s_u and
level driver l_u, n_u the sign indicator of the transitory driver's shock:
    m_{u+1} = clip( 1 - rho_m + rho_m m_u + phi (l_u - h_u) )
    h_{u+1} = clip( m_{u+1} + beta (h_u - m_u) + alpha (s_u - m_u) + gamma (n_u s_u - m_u / 2) )
    h_1 = m_1 = 1,  clip to [.05, 25],  rho_T = alpha + beta + gamma/2 < rho_m < 1.
Drivers are either the issuer's own standardized squared total return z^2/v
('total') or its standardized squared private residual e^2/d ('residual'). With
phi = 0 the level stays at one and the recursion with the total driver is exactly
the bounded GJR law accepted throughout this research:
    h_{u+1} = clip( 1 - rho_T + beta h_u + (alpha + gamma n_u) s_u ).
The five coefficients are pooled across issuers at a formation date and estimated
by Gaussian quasi-likelihood on standardized residual squares r = e^2 / d; the
scale d is profiled.
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

from civol.core import adaptive_garch as garch   # accepted variance-targeted GARCH(1,1) QML for factor series

H_LOW, H_HIGH = .05, 25.
MAX_PERSISTENCE = .995
# point = (rho_T, share, asym, w, phi); alpha = rho_T share (1-asym), beta = rho_T (1-share), gamma = 2 rho_T share asym,
# rho_m = 1 - (1 - rho_T) w, so w in (0,1) keeps rho_T < rho_m < 1.
STARTS = ((.90, .10, .25, .10, .010), (.80, .30, .25, .05, .020), (.95, .05, .50, .20, .005))
BOUNDS_COMPONENT = ((0., MAX_PERSISTENCE), (0., 1.), (0., 1.), (.002, .90), (0., .20))
BOUNDS_NESTED = ((0., MAX_PERSISTENCE), (0., 1.), (0., 1.), (.5, .5), (0., 0.))
DRIVERS = ('total', 'residual')


# ----------------------------------------------------------------------------- coefficients
def coefficients(point):
    rho, share, asym, w, phi = point
    kappa = rho * share
    return np.array([kappa * (1 - asym), rho * (1 - share), 2 * kappa * asym, 1 - (1 - rho) * w, phi])   # alpha, beta, gamma, rho_m, phi


def coefficient_jacobian(point):
    rho, share, asym, w, phi = point
    return np.array([[share * (1 - asym), rho * (1 - asym), -rho * share, 0., 0.],
                     [1 - share, -rho, 0., 0., 0.],
                     [2 * share * asym, 2 * rho * asym, 2 * rho * share, 0., 0.],
                     [w, 0., 0., -(1 - rho), 0.],
                     [0., 0., 0., 0., 1.]])


def point_of(coeff):
    alpha, beta, gamma, rho_m, phi = coeff
    rho = alpha + beta + .5 * gamma
    kappa = alpha + .5 * gamma
    share = kappa / rho if rho > 0 else .3
    asym = .5 * gamma / kappa if kappa > 0 else .25
    w = (1 - rho_m) / (1 - rho) if rho < 1 else .5
    return np.array([rho, share, asym, min(max(w, .002), .9), phi])


def pad_coefficients(c, N):
    """Accept (3,), (5,), (N,3) or (N,5); return N x 5 (nested law when 3 given)."""
    c = np.asarray(c, float)
    if c.ndim == 1:
        c = np.broadcast_to(c, (N, len(c))).copy()
    if c.shape[1] == 3:
        c = np.column_stack([c, np.full(N, .999), np.zeros(N)])
    if c.shape != (N, 5):
        raise ValueError('coefficients must be N x 5')
    return np.ascontiguousarray(c)


# ----------------------------------------------------------------------------- kernels
@njit(parallel=True, cache=True)
def _pooled_objective(alpha, beta, gamma, rho_m, phi, st, nt, sl, response):
    T, N = st.shape
    obj = np.zeros(N); grad = np.zeros((N, 5))
    for i in prange(N):
        h = 1.0; m = 1.0
        dh = np.zeros(5); dm = np.zeros(5)
        o = 0.0; g = np.zeros(5)
        for t in range(T):
            if t > 0:
                s = st[t - 1, i]; l = sl[t - 1, i]
                n = 1.0 if nt[t - 1, i] else 0.0
                pm = 1.0 - rho_m + rho_m * m + phi * (l - h)
                ndm = np.empty(5)
                for j in range(5):
                    ndm[j] = rho_m * dm[j] - phi * dh[j]
                ndm[3] += m - 1.0
                ndm[4] += l - h
                if pm <= H_LOW:
                    pm = H_LOW
                    for j in range(5):
                        ndm[j] = 0.0
                elif pm >= H_HIGH:
                    pm = H_HIGH
                    for j in range(5):
                        ndm[j] = 0.0
                ph = pm + beta * (h - m) + alpha * (s - m) + gamma * (n * s - .5 * m)
                ndh = np.empty(5)
                for j in range(5):
                    ndh[j] = ndm[j] + beta * (dh[j] - dm[j]) - alpha * dm[j] - .5 * gamma * dm[j]
                ndh[0] += s - m
                ndh[1] += h - m
                ndh[2] += n * s - .5 * m
                if ph <= H_LOW:
                    ph = H_LOW
                    for j in range(5):
                        ndh[j] = 0.0
                elif ph >= H_HIGH:
                    ph = H_HIGH
                    for j in range(5):
                        ndh[j] = 0.0
                h = ph; m = pm
                for j in range(5):
                    dh[j] = ndh[j]; dm[j] = ndm[j]
            r = response[t, i]
            o += np.log(h) + r / h
            mult = (1.0 - r / h) / h
            for j in range(5):
                g[j] += mult * dh[j]
        obj[i] = o
        for j in range(5):
            grad[i, j] = g[j]
    scale = 1.0 / (T * N)
    out = np.empty(6)
    out[0] = obj.sum() * scale
    for j in range(5):
        out[j + 1] = grad[:, j].sum() * scale
    return out


@njit(parallel=True, cache=True)
def _variance_path(st, nt, sl, coeff):
    """h and m paths (T x N), next states (N), clip counts; coeff is N x 5."""
    T, N = st.shape
    path = np.empty((T, N)); level = np.empty((T, N)); nxt = np.empty(N); mnxt = np.empty(N)
    lower = 0; upper = 0
    for i in prange(N):
        alpha = coeff[i, 0]; beta = coeff[i, 1]; gamma = coeff[i, 2]; rho_m = coeff[i, 3]; phi = coeff[i, 4]
        h = 1.0; m = 1.0
        for t in range(T):
            path[t, i] = h; level[t, i] = m
            s = st[t, i]; l = sl[t, i]
            n = 1.0 if nt[t, i] else 0.0
            pm = 1.0 - rho_m + rho_m * m + phi * (l - h)
            if pm <= H_LOW:
                pm = H_LOW; lower += 1
            elif pm >= H_HIGH:
                pm = H_HIGH; upper += 1
            ph = pm + beta * (h - m) + alpha * (s - m) + gamma * (n * s - .5 * m)
            if ph <= H_LOW:
                ph = H_LOW; lower += 1
            elif ph >= H_HIGH:
                ph = H_HIGH; upper += 1
            h = ph; m = pm
        nxt[i] = h; mnxt[i] = m
    return path, level, nxt, mnxt, lower, upper


@njit(parallel=True, cache=True)
def _garch_per_issuer(alpha, beta, squares):
    """Variance-targeted GARCH(1,1) QML objective per issuer; squares normalized."""
    T, N = squares.shape
    obj = np.empty(N); last = np.empty(N)
    for i in prange(N):
        a = alpha[i]; b = beta[i]
        h = 1.0; o = 0.0
        for t in range(T):
            if t > 0:
                h = 1.0 - a - b + a * squares[t - 1, i] + b * h
                if h < 1e-8:
                    h = 1e-8
            o += np.log(h) + squares[t, i] / h
        obj[i] = .5 * o / T
        last[i] = 1.0 - a - b + a * squares[T - 1, i] + b * h
    return obj, last


# ----------------------------------------------------------------------------- drivers
def make_drivers(z, residuals, d, reference, transitory='total', level='residual'):
    """Standardized driver panels. Returns (st, nt, sl)."""
    if transitory not in DRIVERS or level not in DRIVERS:
        raise ValueError('drivers must be total or residual')
    z = np.asarray(z, float); e = np.asarray(residuals, float)
    s_total = np.ascontiguousarray(z * z / reference[None, :]); n_total = np.ascontiguousarray(z < 0)
    r = np.ascontiguousarray(e * e / d[None, :]); n_res = np.ascontiguousarray(e < 0)
    st, nt = (s_total, n_total) if transitory == 'total' else (r, n_res)
    sl = s_total if level == 'total' else r
    return st, nt, sl


# ----------------------------------------------------------------------------- pooled fit
def pooled_objective(point, st, nt, sl, response):
    alpha, beta, gamma, rho_m, phi = coefficients(point)
    out = _pooled_objective(alpha, beta, gamma, rho_m, phi, st, nt, sl, response)
    return float(out[0]), coefficient_jacobian(point).T @ out[1:]


def fit_pooled_coefficients(st, nt, sl, response, component=True, extra_starts=(), w_bounds=None):
    """Fixed starts plus optional warm starts; the static point is always a candidate.
    component=False pins phi = 0 (the nested one-component law). w_bounds overrides
    the level-persistence bounds (w = (1 - rho_m) / (1 - rho_T))."""
    st = np.ascontiguousarray(st, dtype=np.float64); nt = np.ascontiguousarray(nt, dtype=np.bool_)
    sl = np.ascontiguousarray(sl, dtype=np.float64); response = np.ascontiguousarray(response, dtype=np.float64)
    bounds = list(BOUNDS_COMPONENT if component else BOUNDS_NESTED)
    if w_bounds is not None and component:
        bounds[3] = tuple(w_bounds)
    bounds = tuple(bounds)
    static_point = np.array([0., .3, .25, .5, 0.])
    static = float(_pooled_objective(0., 0., 0., .5, 0., st, nt, sl, response)[0])
    best = dict(coefficients=coefficients(static_point), objective=static, converged=True, iterations=0, static_selected=True, point=static_point)
    records = []
    for start in list(STARTS) + [tuple(s) for s in extra_starts]:
        x0 = np.array(start, float)
        if not component:
            x0[3] = .5; x0[4] = 0.
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(pooled_objective, x0, args=(st, nt, sl, response), jac=True, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=200, ftol=1e-11, gtol=1e-7, maxls=40))
        records.append(dict(start=x0.tolist(), success=bool(res.success), iterations=int(res.nit), objective=float(res.fun), point=res.x.tolist()))
        if np.isfinite(res.fun) and np.isfinite(res.x).all() and res.fun < best['objective']:
            best = dict(coefficients=coefficients(res.x), objective=float(res.fun), converged=bool(res.success), iterations=int(res.nit),
                        static_selected=False, point=res.x.copy())
    best['optimizer_records'] = records
    best['optimizer_failures'] = int(sum(not r['success'] for r in records))
    best['component'] = bool(component)
    return best


def fit_private_dynamics(z, residuals, d0, reference, alternations=2, component=True, transitory='total', level='residual', w_bounds=None):
    """Pooled coefficients (shared across issuers) and profiled scales.

    z: T x N demeaned total log returns; residuals: T x N private residuals;
    d0: initial scales; reference: v_i = ||b_i||^2 + d_i (total reference variance).
    The component fit is warm-started from the nested one-component solution, so
    its quasi-likelihood is never worse than the nested law's.
    """
    z = np.asarray(z, float); e = np.asarray(residuals, float)
    T, N = z.shape
    d = np.asarray(d0, float).copy()
    history = []
    nested_fit = None
    for it in range(alternations):
        st, nt, sl = make_drivers(z, e, d, reference, transitory, level)
        response = np.ascontiguousarray(e * e / d[None, :])
        nested_fit = fit_pooled_coefficients(st, nt, sl, response, component=False)
        if component:
            p = nested_fit['point']
            extra = ((p[0], p[1], p[2], .5, 0.), (p[0], p[1], p[2], .2, .01), (p[0], p[1], p[2], .05, .02))
            fit = fit_pooled_coefficients(st, nt, sl, response, component=True, extra_starts=extra, w_bounds=w_bounds)
            if fit['objective'] > nested_fit['objective'] and w_bounds is None:
                fit = dict(nested_fit, component=True)
        else:
            fit = nested_fit
        c = fit['coefficients']
        path, level_path, nxt, mnxt, lower, upper = _variance_path(st, nt, sl, pad_coefficients(c, N))
        d = np.maximum(np.mean(e * e / path, axis=0), 1e-12)
        history.append(dict(coefficients=c.tolist(), objective=fit['objective'], nested_objective=nested_fit['objective'], converged=fit['converged'],
                            static_selected=fit['static_selected'], iterations=fit['iterations'], optimizer_failures=fit['optimizer_failures'],
                            lower_contacts=int(lower), upper_contacts=int(upper)))
    return dict(coefficients=c, d=d, h_path=path, m_path=level_path, h_next=nxt, m_next=mnxt, objective=fit['objective'],
                nested_objective=nested_fit['objective'], nested_coefficients=nested_fit['coefficients'], converged=fit['converged'],
                static_selected=fit['static_selected'], history=history, lower_contacts=int(lower), upper_contacts=int(upper),
                optimizer_records=fit['optimizer_records'], component=bool(component), transitory_driver=transitory, level_driver=level)


def variance_path(z, residuals, d, reference, coefficients_matrix, transitory='total', level='residual'):
    N = np.asarray(z).shape[1]
    st, nt, sl = make_drivers(z, residuals, d, reference, transitory, level)
    return _variance_path(st, nt, sl, pad_coefficients(coefficients_matrix, N))


# ----------------------------------------------------------------------------- comparators
def garch_per_issuer(residuals, horizon=21):
    """Variance-targeted GARCH(1,1) per issuer by vectorized grid search with local
    refinement. Returns the horizon sum of expected variances and parameters."""
    e = np.asarray(residuals, float); T, N = e.shape
    target = np.mean(e * e, axis=0)
    squares = np.ascontiguousarray(e * e / target[None, :])
    rhos = np.r_[0., np.linspace(.50, .995, 24)]
    shares = np.linspace(.02, .40, 20)
    best_obj = np.full(N, np.inf); best_a = np.zeros(N); best_b = np.zeros(N)
    for rho in rhos:
        for share in (shares if rho > 0 else [0.]):
            a = np.full(N, rho * share); b = np.full(N, rho * (1 - share))
            obj, _ = _garch_per_issuer(a, b, squares)
            better = obj < best_obj
            best_obj[better] = obj[better]; best_a[better] = a[better]; best_b[better] = b[better]
    step_rho, step_share = (rhos[2] - rhos[1]) / 2, (shares[1] - shares[0]) / 2
    for _ in range(4):
        rho0 = best_a + best_b
        share0 = np.where(rho0 > 0, best_a / np.maximum(rho0, 1e-12), 0.)
        for dr in (-1, 0, 1):
            for ds in (-1, 0, 1):
                if dr == 0 and ds == 0:
                    continue
                rho = np.clip(rho0 + dr * step_rho, 0., MAX_PERSISTENCE)
                share = np.clip(share0 + ds * step_share, 0., 1.)
                a = rho * share; b = rho * (1 - share)
                obj, _ = _garch_per_issuer(a, b, squares)
                better = obj < best_obj
                best_obj[better] = obj[better]; best_a[better] = a[better]; best_b[better] = b[better]
        step_rho /= 2; step_share /= 2
    _, last = _garch_per_issuer(best_a, best_b, squares)
    rho = best_a + best_b
    total = np.zeros(N); h = last.copy()
    for _ in range(horizon):
        total += h
        h = 1. + rho * (h - 1.)
    return dict(forecast=total * target, alpha=best_a, beta=best_b, target=target, next_variance=last * target, objective=best_obj)


def ewma_forecast(residuals, horizon=21, decay=.94):
    e = np.asarray(residuals, float); T, N = e.shape
    v = np.mean(e[:50] * e[:50], axis=0) if T >= 50 else np.mean(e * e, axis=0)
    for t in range(T):
        v = decay * v + (1 - decay) * e[t] * e[t]
    return horizon * v


def realized_1m(residuals, horizon=21, window=21):
    e = np.asarray(residuals, float)[-window:]
    return horizon * np.mean(e * e, axis=0)


def factor_garch(F):
    """Accepted variance-targeted GARCH(1,1) for each factor score series."""
    fits = [garch.fit_series(F[:, k]) for k in range(F.shape[1])]
    omega = np.array([f['target'] * (1 - f['alpha'] - f['beta']) for f in fits])
    alpha = np.array([f['alpha'] for f in fits]); beta = np.array([f['beta'] for f in fits])
    g_next = np.array([f['next_variance'] for f in fits]); target = np.array([f['target'] for f in fits])
    return dict(omega=omega, alpha=alpha, beta=beta, g_next=g_next, target=target,
                converged=int(sum(f['converged'] for f in fits)), optimizer_failures=int(sum(f['optimizer_failures'] for f in fits)),
                static_selected=int(sum(f['static_selected'] for f in fits)))
