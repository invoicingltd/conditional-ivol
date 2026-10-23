"""Paired-twin horizon integration and its deterministic surrogate.

Two payoff branches share the complete future common path (factor draws and
factor variances) and receive independent private innovations; each branch
updates its own private variance states (h, m) with the bounded two-component
recursion of private_dynamics.py, using the configured drivers (total shock or
private shock, each standardized). The final day's common factor and both
private shocks are integrated analytically. Reports q (compounded private payoff
variance), IPV (integrated private variance of the 21-day log-residual sum),
total payoff mean and variance, and batch Monte Carlo standard errors.

backend='numpy' is the reference implementation; backend='gpu' runs the same
law in horizon_gpu.py under the torch environment and falls back to numpy if
the GPU call fails (the fallback is recorded).
"""
from __future__ import annotations
import os, subprocess, tempfile
from pathlib import Path
import numpy as np
from civol.core.paths import GPU_PYTHON, HERE

H_LOW, H_HIGH = .05, 25.
DRIVER_CODE = dict(total=0, residual=1)


def _update(total_shock, eps, h, m, coeff, reference, drivers):
    """One recursion step. eps are the standard-normal private innovations, so the
    standardized private driver is h * eps^2 (its sign is that of eps)."""
    alpha, beta, gamma, rho_m, phi = (coeff[:, j] for j in range(5))
    s_total = total_shock * total_shock / reference
    r = h * eps * eps
    if drivers[0] == 0:
        s, neg = s_total, total_shock < 0
    else:
        s, neg = r, eps < 0
    l = s_total if drivers[1] == 0 else r
    pm = 1 - rho_m + rho_m * m + phi * (l - h)
    lower = int(np.count_nonzero(pm <= H_LOW)); upper = int(np.count_nonzero(pm >= H_HIGH))
    pm = np.clip(pm, H_LOW, H_HIGH)
    ph = pm + beta * (h - m) + alpha * (s - m) + gamma * (neg * s - .5 * m)
    lower += int(np.count_nonzero(ph <= H_LOW)); upper += int(np.count_nonzero(ph >= H_HIGH))
    return np.clip(ph, H_LOW, H_HIGH), pm, lower, upper


def _batch(model, horizon, count, rng):
    mu = model['mu']; b = model['loadings']; d = model['d']; v = model['reference']
    coeff = model['coefficients']; drivers = model['drivers']; N = len(mu); K = b.shape[1]
    omega, af, bf, g_bound = model['factor_omega'], model['factor_alpha'], model['factor_beta'], model['g_bound']
    g = np.broadcast_to(np.minimum(model['g_next'], g_bound), (count, K)).copy()
    ha = np.broadcast_to(model['h_next'], (count, N)).copy(); hb = ha.copy()
    ma_ = np.broadcast_to(model['m_next'], (count, N)).copy(); mb_ = ma_.copy()
    pa = np.zeros((count, N)); pb = np.zeros((count, N))
    ipv = d[None, :] * (ha + hb) * .5          # day 1 private variance (both branches)
    lower = upper = 0; g_contacts = 0
    for day in range(horizon - 1):
        f = np.sqrt(g) * rng.standard_normal((count, K))
        g = omega + af * f * f + bf * g
        g_contacts += int(np.count_nonzero(g >= g_bound)); g = np.minimum(g, g_bound)   # bounded common variance (finite moments)
        common = f @ b.T
        ea = rng.standard_normal((count, N)); eb = rng.standard_normal((count, N))
        sa = common + np.sqrt(ha * d) * ea
        sb = common + np.sqrt(hb * d) * eb
        pa += mu + sa; pb += mu + sb
        ha, ma_, la, ua = _update(sa, ea, ha, ma_, coeff, v, drivers); hb, mb_, lb, ub = _update(sb, eb, hb, mb_, coeff, v, drivers)
        lower += la + lb; upper += ua + ub
        ipv += d[None, :] * (ha + hb) * .5
    c = ((b * b) @ g.T).T                       # count x N common variance on the final day
    da = ha * d; db = hb * d
    ma = np.exp(pa + mu + .5 * da); mb = np.exp(pb + mu + .5 * db)
    e2c = np.exp(2 * c)
    q = .5 * e2c * ((ma - mb) ** 2 + ma * ma * np.expm1(da) + mb * mb * np.expm1(db))
    mean = np.exp(.5 * c) * .5 * (ma + mb)
    second = .5 * e2c * (ma * ma * np.exp(da) + mb * mb * np.exp(db))
    return dict(q=q.mean(0), ipv=ipv.mean(0), mean=mean.mean(0), second=second.mean(0), lower=lower, upper=upper, g_contacts=g_contacts)


G_BOUND_MULTIPLE = 25.   # common-factor variance bound: this multiple of the factor's unconditional (formation-window) variance, the
                         # same multiple at which the private state is bounded relative to its reference variance. Applied to the
                         # formation state and to every future day, it keeps every exponential moment of the compounded payoff finite
                         # and removes the explosive common-variance paths that otherwise dominate q at crisis dates.


def _prepare(model):
    model = dict(model)
    N = len(np.asarray(model['mu']))
    omega = np.asarray(model['factor_omega'], float); af = np.asarray(model['factor_alpha'], float); bf = np.asarray(model['factor_beta'], float)
    target = np.asarray(model['factor_target'], float) if model.get('factor_target') is not None else omega / np.maximum(1 - af - bf, 1e-12)
    g_next = np.asarray(model['g_next'], float)
    model['g_bound'] = np.ascontiguousarray(float(model.get('factor_bound_multiple', G_BOUND_MULTIPLE)) * target)
    model['g_next_clipped_factors'] = int(np.sum(g_next > model['g_bound']))
    model['g_next'] = np.minimum(g_next, model['g_bound'])
    c = np.asarray(model['coefficients'], float)
    if c.ndim == 1:
        c = np.broadcast_to(c, (N, len(c))).copy()
    if c.shape[1] == 3:
        c = np.column_stack([c, np.full(N, .999), np.zeros(N)])
    model['coefficients'] = np.ascontiguousarray(c)
    if model.get('m_next') is None:
        model['m_next'] = np.ones(N)
    drivers = model.get('drivers', ('total', 'residual'))
    model['drivers'] = np.array([DRIVER_CODE[drivers[0]], DRIVER_CODE[drivers[1]]], dtype=np.int64)
    for key in ('mu', 'loadings', 'd', 'reference', 'h_next', 'm_next', 'factor_omega', 'factor_alpha', 'factor_beta', 'g_next'):
        model[key] = np.ascontiguousarray(np.asarray(model[key], float))
    if model['loadings'].ndim == 1:
        model['loadings'] = model['loadings'][:, None]
    return model


def _numpy_batches(model, horizon, pairs, batches, seed):
    pieces = []
    with np.errstate(over='raise', invalid='raise'):
        for batch in range(batches):
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), batch]))
            pieces.append(_batch(model, horizon, pairs // batches, rng))
    stack = {k: np.stack([p[k] for p in pieces]) for k in ('q', 'ipv', 'mean', 'second')}
    return stack, int(sum(p['lower'] for p in pieces)), int(sum(p['upper'] for p in pieces)), dict(backend='numpy', common_upper_contacts=int(sum(p['g_contacts'] for p in pieces)))


def _gpu_batches(model, horizon, pairs, batches, seed):
    if not Path(GPU_PYTHON).is_file():
        raise FileNotFoundError(GPU_PYTHON)
    tmp = Path(tempfile.mkdtemp(prefix='civol_gpu_'))
    try:
        fin = tmp / 'model.npz'; fout = tmp / 'result.npz'
        np.savez(fin, horizon=horizon, pairs=pairs, batches=batches, seed=seed, **{k: model[k] for k in
                 ('mu', 'loadings', 'd', 'reference', 'h_next', 'm_next', 'coefficients', 'drivers', 'factor_omega', 'factor_alpha', 'factor_beta', 'g_next', 'g_bound')})
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE')
        res = subprocess.run([str(GPU_PYTHON), '-I', str(HERE / 'horizon_gpu.py'), str(fin), str(fout)], capture_output=True, text=True, timeout=1800, env=env)
        if res.returncode != 0 or not fout.is_file():
            raise RuntimeError('GPU integration failed: ' + res.stderr[-2000:])
        r = np.load(fout)
        stack = {k: np.asarray(r[k], float) for k in ('q', 'ipv', 'mean', 'second')}
        info = dict(backend='gpu', device=str(r['device']), torch_version=str(r['torch_version']), common_upper_contacts=int(r['g_contacts']) if 'g_contacts' in r.files else None)
        return stack, int(r['lower']), int(r['upper']), info
    finally:
        for f in tmp.glob('*'):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass


def paired_moments(model, horizon=21, pairs=8192, batches=8, seed=0, backend='numpy'):
    """model keys: mu, loadings (N x K, return units), d, reference, h_next, m_next,
    coefficients (3, 5, N x 3 or N x 5), drivers (transitory, level),
    factor_omega/alpha/beta, g_next (K)."""
    if pairs % batches or pairs // batches < 2 or batches < 2:
        raise ValueError('pairs must split into at least two equal batches of at least two pairs')
    model = _prepare(model)
    if backend == 'gpu':
        try:
            stack, lower, upper, info = _gpu_batches(model, horizon, pairs, batches, seed)
        except Exception as exc:
            stack, lower, upper, info = _numpy_batches(model, horizon, pairs, batches, seed)
            info = dict(backend='numpy_fallback', gpu_error=f'{type(exc).__name__}: {exc}'[:500])
    elif backend == 'numpy':
        stack, lower, upper, info = _numpy_batches(model, horizon, pairs, batches, seed)
    else:
        raise ValueError(backend)
    q = stack['q'].mean(0); ipv = stack['ipv'].mean(0); mean = stack['mean'].mean(0)
    total_variance = stack['second'].mean(0) - mean * mean
    batch_total = stack['second'] - stack['mean'] ** 2
    se = lambda a: a.std(0, ddof=1) / np.sqrt(batches)
    out = dict(q=q, ipv=ipv, mean_return=mean - 1, total_variance=total_variance,
               q_mcse=se(stack['q']), ipv_mcse=se(stack['ipv']), total_variance_mcse=se(batch_total), mean_mcse=se(stack['mean']),
               lower_contacts=int(lower), upper_contacts=int(upper), pairs=int(pairs), batches=int(batches), seed=int(seed),
               g_next_clipped_factors=model['g_next_clipped_factors'], factor_bound_multiple=float(model.get('factor_bound_multiple', G_BOUND_MULTIPLE)), **info)
    if not all(np.isfinite(out[k]).all() for k in ('q', 'ipv', 'total_variance')) or np.any(q <= 0) or np.any(total_variance <= 0):
        raise FloatingPointError('Invalid paired forecast')
    return out


def expected_variance_paths(model, horizon=21):
    """Deterministic expected private and common variance sums (surrogate).

    Ignores clipping; E[total driver] = (sum_k b_k^2 E g_k + d E h) / v and
    E[residual driver] = E h.
    """
    model = _prepare(model)
    mu = model['mu']; b = model['loadings']; d = model['d']; v = model['reference']
    coeff = model['coefficients']; drivers = model['drivers']
    alpha, beta, gamma, rho_m, phi = (coeff[:, j] for j in range(5))
    omega, af, bf, g_bound = model['factor_omega'], model['factor_alpha'], model['factor_beta'], model['g_bound']
    g = np.minimum(model['g_next'], g_bound); h = model['h_next'].copy(); m = model['m_next'].copy()
    pv = np.zeros(len(mu)); cv = np.zeros(len(mu))
    for _ in range(horizon):
        common = (b * b) @ g
        pv += d * h; cv += common
        es_total = (common + d * h) / v
        es = es_total if drivers[0] == 0 else h
        el = es_total if drivers[1] == 0 else h
        pm = np.clip(1 - rho_m + rho_m * m + phi * (el - h), H_LOW, H_HIGH)
        h = np.clip(pm + beta * (h - m) + alpha * (es - m) + gamma * (.5 * es - .5 * m), H_LOW, H_HIGH)
        m = pm
        g = np.minimum(omega + (af + bf) * g, g_bound)
    return pv, cv


def surrogate_moments(model, horizon=21):
    pv, cv = expected_variance_paths(model, horizon)
    hm = horizon * np.asarray(model['mu'], float)
    q = np.exp(2 * hm + 2 * cv + pv) * np.expm1(pv)
    total = np.exp(2 * hm + cv + pv) * np.expm1(cv + pv)
    mean = np.exp(hm + .5 * (cv + pv)) - 1
    return dict(q=q, ipv=pv, common_variance_sum=cv, total_variance=total, mean_return=mean)
