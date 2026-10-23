"""Generating laws with known truth for the simulation study.

Daily log returns: x_iu = mu_i + b_i' f_u + sqrt(d_i h_iu) eps_iu, with K0 = 5
GARCH(1,1) factors and the bounded two-component private law of
core/private_dynamics.py (pooled coefficients, issuer-specific scales). The
observed "FF3" factors are the first three true factors. Truth q_i and the true
integrated private variance are the paired functional under the true law and
the true formation states.
"""
from __future__ import annotations
import numpy as np
from civol.core import horizon as hz

CELLS = ('baseline', 'heterogeneous_feedback', 'weak_directions', 'block_private')
K0 = 5
FACTOR_ALPHA = np.array([.06, .05, .04, .03, .03])
FACTOR_BETA = np.array([.92, .93, .94, .95, .95])
POOLED = np.array([.04, .88, .04, .995, .02])      # alpha, beta, gamma, rho_m, phi
H_LOW, H_HIGH = .05, 25.
G_BOUND = 25.                                        # factor variance bound (unit unconditional variance)


def draw_parameters(cell, rng, N):
    s = np.exp(rng.normal(np.log(.02), .4, N))                 # total daily volatility
    lam = np.column_stack([rng.normal(.55, .15, N), rng.normal(0, .30, N), rng.normal(0, .30, N),
                           rng.normal(0, .20, N), rng.normal(0, .20, N)])
    if cell == 'weak_directions':
        lam[:, 3:] *= .25
    norm2 = np.sum(lam * lam, 1)
    too_big = norm2 > .8
    lam[too_big] *= np.sqrt(.8 / norm2[too_big])[:, None]
    psi = 1 - np.sum(lam * lam, 1)                             # private share of standardized variance
    b = lam * s[:, None]
    d = psi * s * s
    mu = rng.normal(3e-4, 2e-4, N)
    if cell == 'heterogeneous_feedback':
        rho = rng.uniform(.80, .98, N); share = rng.uniform(.03, .20, N); asym = rng.uniform(.1, .7, N)
        kappa = rho * share
        rho_m = rng.uniform(.985, .999, N); phi = rng.uniform(.005, .04, N)
        coeff = np.column_stack([kappa * (1 - asym), rho * (1 - share), 2 * kappa * asym, rho_m, phi])
    else:
        coeff = np.broadcast_to(POOLED, (N, 5)).copy()
    groups = np.arange(N) // 8 if cell == 'block_private' else None
    return dict(mu=mu, loadings=b, d=d, s=s, coefficients=coeff, groups=groups, private_corr=.3 if groups is not None else 0.)


def simulate(cell, seed, N=2000, T=750, H=21, burn=750, transitory='residual', level='total'):
    if cell not in CELLS:
        raise ValueError(cell)
    rng = np.random.default_rng(np.random.SeedSequence([20261020, CELLS.index(cell), seed]))
    p = draw_parameters(cell, rng, N)
    mu, b, d, coeff = p['mu'], p['loadings'], p['d'], p['coefficients']
    v = np.sum(b * b, 1) + d
    alpha, beta, gamma, rho_m, phi = (coeff[:, j] for j in range(5))
    omega = 1 - FACTOR_ALPHA - FACTOR_BETA                      # unit unconditional factor variance
    g = np.ones(K0); h = np.ones(N); m = np.ones(N)
    total = burn + T + H
    x = np.empty((total, N)); F = np.empty((total, K0))
    h_record = None; m_record = None; g_record = None
    for u in range(total):
        f = np.sqrt(g) * rng.standard_normal(K0)
        eps = rng.standard_normal(N)
        if p['groups'] is not None:
            shared = rng.standard_normal(p['groups'].max() + 1)[p['groups']]
            eps = np.sqrt(p['private_corr']) * shared + np.sqrt(1 - p['private_corr']) * eps
        z = b @ f + np.sqrt(d * h) * eps
        x[u] = mu + z; F[u] = f
        g = np.minimum(omega + FACTOR_ALPHA * f * f + FACTOR_BETA * g, G_BOUND)     # bounded common variance (finite moments)
        s_total = z * z / v; r = h * eps * eps
        s, neg = (s_total, z < 0) if transitory == 'total' else (r, eps < 0)
        l = s_total if level == 'total' else r
        pm = np.clip(1 - rho_m + rho_m * m + phi * (l - h), H_LOW, H_HIGH)
        h = np.clip(pm + beta * (h - m) + alpha * (s - m) + gamma * (neg * s - .5 * m), H_LOW, H_HIGH)
        m = pm
        if u == burn + T - 1:
            h_record = h.copy(); m_record = m.copy(); g_record = g.copy()
    formation = x[burn:burn + T]; future = x[burn + T:]
    ff_obs = F[burn:burn + T, :3]; ff_future = F[burn + T:, :3]
    truth_model = dict(mu=mu, loadings=b, d=d, reference=v, h_next=h_record, m_next=m_record, coefficients=coeff, drivers=(transitory, level),
                       factor_omega=omega, factor_alpha=FACTOR_ALPHA, factor_beta=FACTOR_BETA, g_next=g_record, factor_target=np.ones(K0), factor_bound_multiple=G_BOUND)
    return dict(cell=cell, seed=seed, x=formation, future=future, ff=ff_obs, ff_future=ff_future, rf=np.zeros(T), rf_future=np.zeros(H),
                truth_model=truth_model, parameters=p, factors_formation=F[burn:burn + T])


def truth(sim, pairs=65536, batches=16, seed=99, backend='numpy'):
    return hz.paired_moments(sim['truth_model'], 21, pairs, batches, seed, backend=backend)
