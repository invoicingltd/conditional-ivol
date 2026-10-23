"""Scientific identities and recovery checks on invented data."""
import numpy as np

from civol.core import adaptive_garch as garch
from civol.core import benchmarks as bm
from civol.core import common_space as cs
from civol.core import horizon as hz
from civol.core import outcomes as oc
from civol.core import private_dynamics as pdyn
from civol.core import universe as uv


def planted_panel(T=600, N=400, K=3, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, K))
    L = rng.normal(0, .4, (N, K)); L[:, 0] += .6
    psi = np.clip(1 - np.sum(L * L, 1), .2, None)
    E = rng.standard_normal((T, N)) * np.sqrt(psi)
    s = np.exp(rng.normal(np.log(.02), .3, N))
    x = (F @ L.T + E) * s + .0003
    return x, F, L, s


def test_pca_identities():
    x, F, L, s = planted_panel()
    mu, sd, Z = cs.standardize(x)
    assert np.allclose(Z.mean(0), 0, atol=1e-10) and np.allclose((Z * Z).mean(0), 1, atol=1e-10)
    w, U = cs.gram_eigen(Z)
    Fh, B, E = cs.pca(Z, 3, (w, U))
    assert np.allclose(Fh.T @ Fh / len(Z), np.eye(3), atol=1e-8)
    assert np.allclose(Z, Fh @ B.T + E, atol=1e-10)
    assert np.allclose(np.sum(B * B, 0), w[:3], atol=1e-8)
    assert abs(w.sum() - Z.shape[1]) < 1e-6


def test_rank_rules_detect_planted_factors():
    x, F, L, s = planted_panel(T=600, N=400, K=3, seed=1)
    out = cs.common_space(x, kmax=10, sign_draws=64)
    assert out['rank'] == 3, out['diagnostics']
    assert out['diagnostics']['icp2_rank'] == 3 and out['diagnostics']['sign_rank'] == 3
    assert out['diagnostics']['eigenvalue_ratio_rank'] == 3
    Fh = out['F']; beta, *_ = np.linalg.lstsq(Fh, F, rcond=None)
    r2 = 1 - np.sum((F - Fh @ beta) ** 2) / np.sum(F ** 2)
    assert r2 > .97


def simulate_feedback(T=750, N=400, coeff=(.05, .88, .05, .999, 0.), seed=2, burn=0, transitory='total', level='residual'):
    rng = np.random.default_rng(seed)
    alpha, beta, gamma, rho_m, phi = coeff
    d = np.exp(rng.normal(np.log(3e-4), .3, N)); b = rng.normal(0, .01, (N, 2)); v = np.sum(b * b, 1) + d
    h = np.ones(N); m = np.ones(N); z = np.empty((T, N)); e = np.empty((T, N))
    for t in range(-burn, T):
        f = rng.standard_normal(2)
        eps = rng.standard_normal(N)
        et = np.sqrt(d * h) * eps
        zt = b @ f + et
        if t >= 0:
            e[t] = et; z[t] = zt
        s_total = zt ** 2 / v; r = h * eps ** 2
        s, neg = (s_total, zt < 0) if transitory == 'total' else (r, eps < 0)
        l = s_total if level == 'total' else r
        pm = np.clip(1 - rho_m + rho_m * m + phi * (l - h), .05, 25)
        h = np.clip(pm + beta * (h - m) + alpha * (s - m) + gamma * (neg * s - .5 * m), .05, 25)
        m = pm
    return z, e, d, v, h, m


def test_pooled_fit_recovers_one_component_coefficients():
    z, e, d, v, h_true, _ = simulate_feedback()
    fit = pdyn.fit_private_dynamics(z, e, np.mean(e * e, 0), v, alternations=2, component=False)
    c = fit['coefficients']
    assert np.allclose(c[:3], (.05, .88, .05), atol=.04), c
    assert c[4] == 0. and np.allclose(fit['m_next'], 1.)
    assert np.corrcoef(fit['h_next'], h_true)[0, 1] > .95
    assert np.median(fit['d'] / d) > .9 and np.median(fit['d'] / d) < 1.1


def test_pooled_fit_recovers_two_component_coefficients():
    true = (.04, .86, .04, .995, .02)
    for transitory, level in (('total', 'residual'), ('total', 'total'), ('residual', 'residual')):
        z, e, d, v, h_true, m_true = simulate_feedback(T=1500, N=600, coeff=true, seed=9, transitory=transitory, level=level)
        fit = pdyn.fit_private_dynamics(z, e, np.mean(e * e, 0), v, alternations=2, component=True, transitory=transitory, level=level)
        c = fit['coefficients']
        assert np.allclose(c[:3], true[:3], atol=.05), (transitory, level, c)
        assert abs(c[3] - true[3]) < .01 and abs(c[4] - true[4]) < .015, (transitory, level, c)
        assert np.corrcoef(fit['h_next'], h_true)[0, 1] > .9
        assert fit['objective'] <= fit['nested_objective'] + 1e-12       # warm start: never worse than the nested law


def test_pooled_gradient_matches_finite_difference():
    z, e, d, v, _, _ = simulate_feedback(T=200, N=50, coeff=(.06, .85, .04, .99, .03))
    st, nt, sl = pdyn.make_drivers(z, e, d, v, 'total', 'residual')
    response = np.ascontiguousarray(e * e / d)
    point = np.array([.9, .2, .3, .2, .02])
    f0, g = pdyn.pooled_objective(point, st, nt, sl, response)
    for j in range(5):
        step = np.zeros(5); step[j] = 1e-6
        fd = (pdyn.pooled_objective(point + step, st, nt, sl, response)[0] - pdyn.pooled_objective(point - step, st, nt, sl, response)[0]) / 2e-6
        assert abs(fd - g[j]) < 1e-5 * max(1, abs(g[j])), (j, fd, g[j])


def test_component_law_nests_accepted_law():
    rng = np.random.default_rng(12); T, N = 300, 20
    z = rng.normal(0, .02, (T, N)); e = z * .8; v = np.full(N, 4e-4); d = np.full(N, 2.6e-4)
    c3 = np.array([.07, .85, .05])
    path, level, nxt, mnxt, lo, up = pdyn.variance_path(z, e, d, v, c3, 'total', 'residual')
    assert np.allclose(level, 1.) and np.allclose(mnxt, 1.)
    h = np.ones(N); rho = c3[0] + c3[1] + .5 * c3[2]
    for t in range(T):
        assert np.allclose(path[t], h)
        h = np.clip(1 - rho + c3[1] * h + (c3[0] + c3[2] * (z[t] < 0)) * z[t] ** 2 / v, .05, 25)
    assert np.allclose(nxt, h)


def test_paired_moments_closed_form_without_feedback():
    rng = np.random.default_rng(3); N, K, H = 30, 2, 21
    mu = rng.normal(0, 3e-4, N); b = rng.normal(0, .008, (N, K)); d = np.exp(rng.normal(np.log(2e-4), .3, N))
    model = dict(mu=mu, loadings=b, d=d, reference=np.sum(b * b, 1) + d, h_next=np.ones(N), coefficients=np.zeros(3),
                 factor_omega=np.ones(K), factor_alpha=np.zeros(K), factor_beta=np.zeros(K), g_next=np.ones(K))
    pm = hz.paired_moments(model, H, pairs=16384, batches=8, seed=1)
    cv = H * np.sum(b * b, 1); pv = H * d
    q_exact = np.exp(2 * H * mu + 2 * cv + pv) * np.expm1(pv)
    assert np.allclose(pm['ipv'], pv, rtol=1e-12)
    assert np.all(np.abs(pm['q'] - q_exact) < 4 * pm['q_mcse'] + 1e-12 * q_exact)
    total_exact = np.exp(2 * H * mu + cv + pv) * np.expm1(cv + pv)
    assert np.all(np.abs(pm['total_variance'] - total_exact) < 4 * pm['total_variance_mcse'] + 1e-9)
    sm = hz.surrogate_moments(model, H)
    assert np.allclose(sm['q'], q_exact) and np.allclose(sm['total_variance'], total_exact)


def test_paired_moments_with_feedback_is_consistent_with_surrogate_scale():
    for drivers in (('total', 'residual'), ('total', 'total'), ('residual', 'residual')):
        z, e, d, v, h, m = simulate_feedback(T=300, N=40, coeff=(.05, .88, .05, .995, .02), transitory=drivers[0], level=drivers[1])
        N = len(d); b = np.zeros((N, 1)) + .005
        model = dict(mu=np.zeros(N), loadings=b, d=d, reference=v, h_next=h, m_next=m, coefficients=np.array([.05, .88, .05, .995, .02]), drivers=drivers,
                     factor_omega=np.array([.1]), factor_alpha=np.array([.05]), factor_beta=np.array([.85]), g_next=np.array([1.]))
        pm = hz.paired_moments(model, 21, pairs=8192, batches=8, seed=5)
        sm = hz.surrogate_moments(model, 21)
        ratio = pm['ipv'] / sm['ipv']
        assert np.all((ratio > .8) & (ratio < 1.25)), (drivers, ratio)
        assert np.all(pm['q'] > 0) and np.all(pm['total_variance'] > pm['q'] * .5)


def test_common_variance_bound_never_binds_at_default_and_binds_when_low():
    rng = np.random.default_rng(21); N, K, H = 40, 3, 21
    mu = np.zeros(N); b = rng.normal(0, .01, (N, K)); d = np.exp(rng.normal(np.log(2e-4), .3, N))
    model = dict(mu=mu, loadings=b, d=d, reference=np.sum(b * b, 1) + d, h_next=np.ones(N), coefficients=np.array([.05, .85, .04, .99, .02]),
                 factor_omega=np.array([.06, .07, .08]), factor_alpha=np.array([.08, .10, .12]), factor_beta=np.array([.86, .83, .80]), g_next=np.array([1.5, 1., .8]))
    default = hz.paired_moments(model, H, pairs=4096, batches=4, seed=2)
    huge = hz.paired_moments(dict(model, factor_bound_multiple=1e12), H, pairs=4096, batches=4, seed=2)
    assert default['common_upper_contacts'] == 0
    assert np.array_equal(default['q'], huge['q']) and np.array_equal(default['total_variance'], huge['total_variance'])   # bound inactive: bit-identical
    tight = hz.paired_moments(dict(model, factor_bound_multiple=1.2), H, pairs=4096, batches=4, seed=2)
    assert tight['common_upper_contacts'] > 0 and not np.array_equal(tight['q'], default['q'])
    assert np.mean(tight['total_variance'] < default['total_variance']) > .9       # truncating the common variance lowers total variance
    # the surrogate uses the same bound: 1.2 x unconditional variance per factor, applied to the formation state as well
    target = np.array([.06, .07, .08]) / (1 - np.array([.08, .10, .12]) - np.array([.86, .83, .80]))
    bound = 1.2 * target
    sm = hz.surrogate_moments(dict(model, factor_bound_multiple=1.2), H)
    assert np.all(sm['common_variance_sum'] <= H * (b * b) @ bound + 1e-12)
    assert tight['g_next_clipped_factors'] == 1 and default['g_next_clipped_factors'] == 0     # g_next = 1.5 exceeds 1.2 x 1 for the first factor


def test_garch_grid_against_accepted_fit():
    rng = np.random.default_rng(4); T = 750
    series = []
    for alpha, beta in ((.05, .9), (.1, .85), (.02, .95), (0., 0.)):
        h = 1.; y = np.empty(T)
        for t in range(T):
            y[t] = np.sqrt(h) * rng.standard_normal() * .02
            h = 1 - alpha - beta + alpha * (y[t] / .02) ** 2 + beta * h
        series.append(y)
    E = np.column_stack(series)
    mine = pdyn.garch_per_issuer(E, 21)
    for j in range(4):
        acc = garch.fit_series(E[:, j])
        assert mine['objective'][j] <= acc['objective'] + 2e-3, (j, mine['objective'][j], acc['objective'])
        acc_sum = garch.expected_variance_sum(acc, 21)
        assert abs(mine['forecast'][j] / acc_sum - 1) < .15, (j, mine['forecast'][j], acc_sum)


def test_ff3_regression_and_realized():
    rng = np.random.default_rng(6); T, N = 300, 20
    ff = rng.normal(0, .01, (T, 3)); rf = np.full(T, 1e-4)
    beta = rng.normal(0, 1, (N, 3)); e = rng.normal(0, .015, (T, N))
    x = np.log1p(rf)[:, None] + .0002 + ff @ beta.T + e
    out = bm.ff3_space(x, ff, rf)
    assert np.allclose(out['betas'], beta, atol=.2)
    assert np.allclose(out['residuals'].T @ np.column_stack([np.ones(T), ff]), 0, atol=1e-8)
    r = bm.ff3_realized(x, ff, rf, 21, 21)
    y = (x - np.log1p(rf)[:, None])[-21:]
    _, res = bm.ols(y, ff[-21:])
    assert np.allclose(r['forecast'], 21 * np.sum(res ** 2, 0) / 17)


def test_outcomes_recover_planted_residual_variance_and_diversifiability():
    rng = np.random.default_rng(7); H, N, K = 21, 500, 2
    s = np.full(N, .02); mu = np.zeros(N)
    B = rng.normal(0, .5, (N, K)); B[:, 0] += .5          # market-like factor with positive mean loading
    f = rng.standard_normal((H, K))
    psi = .5
    e = rng.standard_normal((H, N)) * np.sqrt(psi)
    z = f @ B.T + e
    x = z * s
    valid = np.ones((H, N), bool)
    ff = rng.normal(0, .01, (H, 3)); rf = np.zeros(H)
    res = oc.realized_outcomes(x, valid, ff, rf, mu, s, B, np.zeros(N), np.zeros((N, 3)), H)
    assert res['scored_count'] == N
    assert abs(np.mean(res['rv_broad']) / (H * psi * s[0] ** 2) - 1) < .1
    assert res['diversifiability_broad'] < 3
    assert res['diversifiability_ff3_fixed'] > 20
    valid[3, 5] = False
    res2 = oc.realized_outcomes(x, valid, ff, rf, mu, s, B, np.zeros(N), np.zeros((N, 3)), H)
    assert not res2['scored'][5] and np.isnan(res2['rv_broad'][5]) and res2['scored_count'] == N - 1


class FakeCache:
    def __init__(self):
        T, N = 800, 6
        rng = np.random.default_rng(8)
        self.dates = np.arange('2000-01-01', '2000-01-01', dtype='datetime64[D]')
        self.permnos = np.array([10, 20, 30, 40, 50, 60])
        self.ret = rng.normal(0, .01, (T, N)).astype(np.float32)
        self.valid = np.ones((T, N), bool)
        self.elig = np.ones((T, N), bool)
        self.mktcap = np.tile(np.array([5., 9., 3., 7., 2., 4.], dtype=np.float32), (T, 1))
        self.permco = np.tile(np.array([1, 1, 2, 3, 4, 5], dtype=np.int32), (T, 1))
        self.valid[100, 2] = False
        self.permco[:300, 3] = 9
        self.elig[-1, 4] = False
        self.ret[:, 5] = 0.
        self.valid[:, 5] = True


def test_eligible_universe_rules():
    c = FakeCache()
    idx, info = uv.eligible_universe(c, 799, 750)
    assert c.permnos[idx].tolist() == [20]
    assert info['selected'] == 1
