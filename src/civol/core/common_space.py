"""Diffuse-aggregation common space: standardized-panel PCA and rank rules.

All eigen-decompositions use the T x T Gram matrix Z Z'/T, whose nonzero
eigenvalues equal those of the N x N sample correlation matrix Z'Z/T. This is
exact and makes the 64-draw sign-rank reference affordable for N in the
thousands. The sign-rank rule reproduces overnight_method/allocation_diagnosis/
sign_rank.py (same null, same 64 draws, same 95th percentile order statistic).
"""
from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

SIGN_DRAWS = 64
SIGN_ALPHA = .05
SIGN_SEED = 2026101901
KMAX = 20


def standardize(x):
    """Sample mean, standard deviation (divisor T) and standardized panel."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite two-dimensional panel required')
    mu = x.mean(0)
    z = x - mu
    s = np.sqrt(np.mean(z * z, axis=0))
    if np.any(s * s <= 1e-14):
        raise ValueError('Degenerate coordinate')
    return mu, s, z / s


def gram_eigen(Z):
    """Descending eigenvalues of Z'Z/T (nonzero part) and matching T-vectors."""
    T = Z.shape[0]
    G = (Z @ Z.T) / T
    w, U = np.linalg.eigh(G)
    w = w[::-1].copy(); U = U[:, ::-1].copy()
    w[w < 0] = 0.
    return w, U


def pca(Z, rank, eig=None):
    """Scores F (T x K, F'F/T = I), standardized loadings B (N x K), residuals."""
    T, N = Z.shape
    w, U = gram_eigen(Z) if eig is None else eig
    if not 0 <= rank < min(N, T - 2):
        raise ValueError('rank outside admissible range')
    F = np.sqrt(T) * U[:, :rank]
    B = Z.T @ F / T
    E = Z - F @ B.T
    return F, B, E


def largest_null_eigenvalue(Z, rng):
    """Largest eigenvalue of (Z*S)(Z*S)'/T for one independent sign matrix S."""
    T, N = Z.shape
    signs = rng.integers(0, 2, size=(T, N), dtype=np.int8) * 2 - 1
    Zs = Z * signs
    if T <= 400:
        return float(np.linalg.eigvalsh(Zs @ Zs.T / T)[-1])
    op = LinearOperator((T, T), matvec=lambda v: Zs @ (Zs.T @ v) / T, dtype=np.float64)
    return float(eigsh(op, k=1, which='LA', tol=1e-7, maxiter=5000, return_eigenvectors=False)[0])


def sign_rank(Z, eigenvalues, draws=SIGN_DRAWS, alpha=SIGN_ALPHA, seed=SIGN_SEED):
    rng = np.random.default_rng(int(seed))
    maxima = np.array([largest_null_eigenvalue(Z, rng) for _ in range(draws)])
    counts = np.sum(maxima[:, None] >= eigenvalues[None, :], axis=0)
    pvalues = (1 + counts) / (draws + 1)
    rank = int(np.sum(pvalues <= alpha))
    if draws < 1 / alpha - 1:
        raise ValueError('Too few null draws for any rejection at this alpha')
    index = min(int(np.ceil((1 - alpha) * (draws + 1))) - 1, draws - 1)
    threshold = float(np.sort(maxima)[index])
    return dict(rank=rank, threshold=threshold, null_largest_eigenvalues=maxima.tolist(), draws=draws, alpha=alpha, seed=int(seed))


def bai_ng_icp2(eigenvalues, T, N, kmax=KMAX):
    """Bai-Ng IC_p2 on the standardized panel (trace = N)."""
    kmax = int(min(kmax, N - 1, T - 3))
    total = float(eigenvalues.sum())
    remaining = np.maximum(total - np.r_[0., np.cumsum(eigenvalues[:kmax])], np.finfo(float).tiny) / N
    penalty = (N + T) / (N * T) * np.log(min(N, T))
    criteria = np.log(remaining) + np.arange(kmax + 1) * penalty
    return int(np.argmin(criteria)), criteria.tolist()


def eigenvalue_ratio(eigenvalues, kmax=KMAX):
    """Ahn-Horenstein eigenvalue ratio over k = 1..kmax."""
    lam = np.maximum(eigenvalues[:kmax + 1], np.finfo(float).tiny)
    ratios = lam[:-1] / lam[1:]
    return int(np.argmax(ratios) + 1), ratios.tolist()


def common_space(x, kmax=KMAX, sign_seed=SIGN_SEED, sign_draws=SIGN_DRAWS, rule='icp2', alternative_ranks=(10, 20), variant_rank=None):
    """Full common-space estimate for a complete T x N log-return panel.

    rule: 'icp2' (primary, Bai-Ng IC_p2) or 'sign' (conservative sign-rank).
    Residual content diagnostics are also computed at the other rule's rank and
    at the fixed alternative ranks, without affecting the primary estimate.
    variant_rank: a fixed rank whose full decomposition (scores, loadings,
    residuals) is returned for the fixed-rank comparison.
    """
    mu, s, Z = standardize(x)
    T, N = Z.shape
    w, U = gram_eigen(Z)
    sign = sign_rank(Z, w[:kmax], draws=sign_draws, seed=sign_seed)
    icp2, criteria = bai_ng_icp2(w, T, N, kmax)
    er, ratios = eigenvalue_ratio(w, kmax)
    chosen = icp2 if rule == 'icp2' else sign['rank']
    rank = int(min(chosen, kmax, N - 1, T - 3))
    F, B, E = pca(Z, rank, (w, U))
    df = T / (T - rank - 1)
    E = E * np.sqrt(df)
    loadings = s[:, None] * B                    # return units
    residuals = E * s[None, :]                   # return units
    d0 = np.mean(residuals * residuals, axis=0)
    alternatives = {}
    for k in sorted({sign['rank'], icp2, *alternative_ranks}):
        k = int(min(k, kmax, N - 1, T - 3))
        if k == rank:
            Bk, Ek = B, E
        else:
            _, Bk, Ek = pca(Z, k, (w, U))
            Ek = Ek * np.sqrt(T / (T - k - 1))
        alternatives[k] = dict(B_std=Bk, content=residual_common_content(Ek * s[None, :]))
    variant = None
    if variant_rank is not None:
        kv = int(min(variant_rank, kmax, N - 1, T - 3))
        Fv, Bv, Ev = pca(Z, kv, (w, U))
        Ev = Ev * np.sqrt(T / (T - kv - 1)) * s[None, :]
        variant = dict(rank=kv, F=Fv, B=Bv, loadings=s[:, None] * Bv, residuals=Ev, d0=np.mean(Ev * Ev, axis=0))
    diagnostics = dict(rank=rank, rule=rule, sign_rank=sign['rank'], sign_threshold=sign['threshold'], sign_seed=sign['seed'],
                       icp2_rank=icp2, icp2_criteria=criteria, eigenvalue_ratio_rank=er, top_eigenvalues=w[:30].tolist(),
                       explained_share=float(w[:rank].sum() / N) if rank else 0., residual_df_multiplier=df, T=T, N=N,
                       alternative_rank_content={str(k): v['content'] for k, v in alternatives.items()})
    return dict(mu=mu, s=s, Z=Z, F=F, B=B, loadings=loadings, residuals=residuals, d0=d0, rank=rank,
                diagnostics=diagnostics, null_largest_eigenvalues=sign['null_largest_eigenvalues'],
                alternative_B_std={k: v['B_std'] for k, v in alternatives.items()}, variant=variant)


def residual_common_content(E):
    """Largest eigenvalue of the residual correlation matrix and the equal-weight
    residual portfolio variance ratio (N * Var(mean_i e) / mean_i Var(e))."""
    T, N = E.shape
    Ec = E - E.mean(0)
    sd = np.sqrt(np.mean(Ec * Ec, axis=0))
    sd[sd <= 0] = np.inf
    top = float(gram_eigen(Ec / sd)[0][0])
    portfolio = Ec.mean(1)
    ratio = float(N * np.mean(portfolio * portfolio) / np.mean(Ec * Ec))
    return dict(top_residual_correlation_eigenvalue=top, equal_weight_residual_variance_ratio=ratio)
