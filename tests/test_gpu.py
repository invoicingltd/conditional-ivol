"""Optional CUDA checks against numpy and analytical moments."""
import numpy as np
import pytest

from civol.core import horizon as hz

pytestmark = pytest.mark.usefixtures("gpu_python")


def model_with_feedback(seed=11, N=300, K=3):
    rng = np.random.default_rng(seed)
    mu = rng.normal(2e-4, 2e-4, N); b = rng.normal(0, .008, (N, K)); b[:, 0] += .01
    d = np.exp(rng.normal(np.log(2.5e-4), .5, N)); v = np.sum(b * b, 1) + d
    h = np.exp(rng.normal(0, .4, N)).clip(.05, 25); m = np.exp(rng.normal(0, .2, N)).clip(.05, 25)
    return dict(mu=mu, loadings=b, d=d, reference=v, h_next=h, m_next=m, coefficients=np.array([.08, .80, .06, .995, .02]), drivers=('total', 'residual'),
                factor_omega=np.array([.05, .06, .07]), factor_alpha=np.array([.07, .05, .04]), factor_beta=np.array([.88, .89, .89]),
                g_next=np.array([1.3, .9, 1.1]))


def test_gpu_matches_numpy_within_mcse():
    for drivers in (('total', 'residual'), ('total', 'total'), ('residual', 'residual')):
        m = dict(model_with_feedback(), drivers=drivers)
        g = hz.paired_moments(m, 21, pairs=32768, batches=8, seed=3, backend='gpu')
        assert g['backend'] == 'gpu', g.get('gpu_error')
        n = hz.paired_moments(m, 21, pairs=32768, batches=8, seed=4, backend='numpy')
        for key in ('q', 'ipv', 'total_variance'):
            se = np.sqrt(g[key + '_mcse'] ** 2 + n[key + '_mcse'] ** 2)
            zscore = (g[key] - n[key]) / se
            assert np.mean(np.abs(zscore) < 4) > .99, (drivers, key, np.abs(zscore).max())
            assert abs(np.median(g[key] / n[key]) - 1) < .01, (drivers, key)


def test_gpu_closed_form_without_feedback():
    rng = np.random.default_rng(5); N, K, H = 50, 2, 21
    mu = rng.normal(0, 3e-4, N); b = rng.normal(0, .008, (N, K)); d = np.exp(rng.normal(np.log(2e-4), .3, N))
    model = dict(mu=mu, loadings=b, d=d, reference=np.sum(b * b, 1) + d, h_next=np.ones(N), coefficients=np.zeros(3),
                 factor_omega=np.ones(K), factor_alpha=np.zeros(K), factor_beta=np.zeros(K), g_next=np.ones(K))
    pm = hz.paired_moments(model, H, pairs=16384, batches=8, seed=1, backend='gpu')
    assert pm['backend'] == 'gpu'
    cv = H * np.sum(b * b, 1); pv = H * d
    q_exact = np.exp(2 * H * mu + 2 * cv + pv) * np.expm1(pv)
    assert np.allclose(pm['ipv'], pv, rtol=1e-5)
    assert np.all(np.abs(pm['q'] - q_exact) < 4 * pm['q_mcse'] + 1e-6 * q_exact)
