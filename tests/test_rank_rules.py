"""Rank selection and portfolio identities on invented data."""
import itertools

import numpy as np

from civol.core import common_space as cs
from civol.rank_rule import rank_rules as rr


def planted(T=750, N=600, K=4, mean_loadings=None, scale=.4, noise=1., seed=0, extra_T=0):
    """Standardized-scale panel with K factors, optional nonzero mean loadings
    (equal-weight exposure) and iid noise; returns x (T+extra_T rows), F, L.
    scale may be a per-factor sequence (distinct dispersions separate the
    principal components so that each one maps to a planted factor)."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T + extra_T, K))
    means = np.zeros(K) if mean_loadings is None else np.asarray(mean_loadings, float)
    scales = np.broadcast_to(np.asarray(scale, float), (K,))
    L = rng.normal(0, 1, (N, K)) * scales[None, :] + means[None, :]
    E = rng.standard_normal((T + extra_T, N)) * noise
    s = np.exp(rng.normal(np.log(.02), .3, N))
    x = (F @ L.T + E) * s + .0002
    return x, F, L, s


def test_random_signed_identity_exact_for_balanced_signs():
    rng = np.random.default_rng(1); T, N = 300, 8
    E = rng.standard_normal((T, N)) + 0.8 * rng.standard_normal((T, 1))      # strongly correlated columns
    cols = [c for c in itertools.product([-1., 1.], repeat=N) if sum(c) == 0]  # every balanced sign vector
    S = np.array(cols).T
    st = rr.random_signed_stats(E, S)
    ew = rr.equal_weight_ratio(E)
    assert ew > 3
    assert abs(st['mean'] - (N - ew) / (N - 1)) < 1e-12
    assert abs(st['identity_prediction'] - st['mean']) < 1e-12


def test_equal_weight_ratio_matches_frozen_definition():
    rng = np.random.default_rng(2)
    E = rng.standard_normal((400, 50)) * np.exp(rng.normal(0, .3, 50))
    frozen = cs.residual_common_content(E)['equal_weight_residual_variance_ratio']
    assert abs(rr.equal_weight_ratio(E) - frozen) < 1e-12
    assert abs(rr.portfolio_variance_ratio(E - E.mean(0), np.full(50, 1 / 50)) - frozen) < 1e-12


def test_rule_a_recovers_equal_weight_loaded_factors():
    # three factors with distinct dispersions (separated components) and a small common
    # mean loading each: every component carries equal-weight exposure N m^2 = 6 >> 0.25
    x, F, L, s = planted(K=3, scale=(.8, .5, .3), mean_loadings=(.1, .1, .1), seed=3)
    ins = rr.insample_curves(x, kmax=15)
    k, attainable = rr.rule_a(ins, 1.25)
    assert attainable and k == 3, (k, ins['equal_weight'][:8])
    assert ins['equal_weight'][2] > 1.25 and ins['equal_weight'][3] <= 1.25
    # the same exposure concentrated in one mean-loading direction is absorbed by the
    # first component alone
    x1, *_ = planted(K=4, scale=.4, mean_loadings=(.6, .3, .25, .2), seed=3)
    assert rr.rule_a(rr.insample_curves(x1, kmax=15), 1.25)[0] == 1


def test_rule_a_floor_fallback():
    curve = dict(equal_weight=[3., 2.2, 1.9, 1.75, 1.62, 1.6, 1.58, 1.57, 1.55])
    k, attainable = rr.rule_a(curve, 1.25)           # floor 1.55 > 1.25: bound 1.25 * 1.55 = 1.9375 -> k = 2
    assert not attainable and k == 2
    k2, ok2 = rr.rule_a(dict(equal_weight=[3., 1.3, 1.1, .9]), 1.25)
    assert ok2 and k2 == 2


def test_rule_b_recovers_planted_rank_direction_free():
    x, F, L, s = planted(mean_loadings=(0., 0., 0., 0.), seed=4)      # no equal-weight exposure at all
    ho = rr.heldout_curve(x, split=500, kmax=15)
    k, attained = rr.rule_b(ho, 1.25)
    assert attained and k == 4, (k, ho['ratio'][:8])
    assert min(ho['ratio'][:4]) > 5 and max(np.abs(np.array(ho['ratio'][4:12]) - 1)) < .35
    x3, *_ = planted(K=3, scale=(.8, .5, .3), mean_loadings=(.1, .1, .1), seed=3)
    assert rr.rule_b(rr.heldout_curve(x3, split=500, kmax=15), 1.25) == (3, True)


def test_decomposition_matches_frozen_variant():
    x, F, L, s = planted(T=500, N=300, K=3, seed=5)
    out = cs.common_space(x, kmax=10, sign_draws=64, variant_rank=7)
    mu, sd, Z = cs.standardize(x); eig = cs.gram_eigen(Z)
    dec = rr.decomposition_at(Z, eig, sd, 7)
    for key in ('F', 'B', 'loadings', 'residuals', 'd0'):
        assert np.allclose(dec[key], out['variant'][key], rtol=1e-11, atol=1e-13), key
    assert abs(rr.equal_weight_ratio(dec['residuals']) - out['diagnostics']['alternative_rank_content'][str(out['rank'])]['equal_weight_residual_variance_ratio']) < 1e-9 \
        if out['rank'] == 7 else True


def test_next_direction_out_of_sample_ratio():
    T, N, H = 750, 600, 21
    x, F, L, s = planted(T=T, N=N, K=3, seed=6, extra_T=20 * H)
    form, future = x[:T], x[T:]
    mu, sd, Z = cs.standardize(form); eig = cs.gram_eigen(Z)
    z_future = (future - mu) / sd
    valid = np.ones_like(z_future, bool); scored = np.ones(N, bool)
    ratios = {}
    for K in (2, 3):
        B = rr.decomposition_at(Z, eig, sd, K)['B']
        v = rr.next_direction(Z, eig, sd, K)
        vals = []
        for w in range(20):
            zf = z_future[w * H:(w + 1) * H]
            from civol.core.outcomes import broad_residuals
            e, _ = broad_residuals(zf, valid[:H], B)
            vals.append(rr.portfolio_variance_ratio(e, v, scored))
        ratios[K] = float(np.median(vals))
    assert ratios[2] > 5, ratios          # the omitted third factor is carried by the next direction
    assert .7 < ratios[3] < 1.4, ratios   # nothing left: one up to 21-session noise


def test_heldout_curve_drops_estimation_degenerate_issuers():
    # An issuer with no price change over the estimation part (eligible
    # because its full-window variance is positive) is left out of rule B; the curve
    # equals the one computed without that issuer, and nothing changes otherwise.
    x, F, L, s = planted(K=3, scale=(.8, .5, .3), mean_loadings=(.1, .1, .1), seed=8)
    xd = x.copy(); xd[:500, 0] = .0002                     # flat for the first 500 sessions, moving afterwards
    ho = rr.heldout_curve(xd, split=500, kmax=10)
    ref = rr.heldout_curve(x[:, 1:], split=500, kmax=10)
    assert ho['dropped_degenerate'] == 1 and ho['N'] == x.shape[1] - 1 and ref['dropped_degenerate'] == 0
    assert np.allclose(ho['ratio'], ref['ratio'], rtol=1e-12, atol=1e-14)
    assert rr.rule_b(ho, 1.25) == rr.rule_b(ref, 1.25) == (3, True)


def test_balanced_signs_are_balanced_and_reproducible():
    S = rr.balanced_signs(101, 16, 7)
    assert S.shape == (101, 16) and np.all(np.sum(S == -1, axis=0) == 50) and np.all(np.abs(S) == 1)
    assert np.array_equal(S, rr.balanced_signs(101, 16, 7)) and not np.array_equal(S, rr.balanced_signs(101, 16, 8))
