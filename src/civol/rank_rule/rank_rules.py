"""Formation-information rank rules that target the diversifiability property,
and the residual-portfolio diagnostics used to evaluate them.

Everything here is computed from the formation window only (rules) or from
realized residual matrices handed in by the caller (out-of-sample diagnostics).

Rule A (equal-weight bound): the smallest rank k in 0..KMAX whose in-sample
    equal-weight residual variance ratio (N Var(mean_i e_i) / mean_i Var(e_i),
    one under uncorrelated residuals) is at or below BOUND * max(1, floor),
    where floor is the minimum ratio across the candidate ranks.
Rule B (held-out next direction): split the formation window into an estimation
    part (first SPLIT sessions) and a validation part (the rest). For each k the
    (k+1)-th principal direction of the estimation part is a fixed diffuse
    portfolio; its validation-window residual variance ratio
    (Var(v' e_k) / sum_i v_i^2 Var(e_{k,i}), e_k the k-factor validation residuals)
    is one when no common variation remains along it. Rule B is the smallest k
    whose held-out ratio is at or below BOUND.
Random-signed portfolios (balanced +-1/N weights): their expected variance ratio
    equals (N - EW)/(N - 1) for any residual covariance (EW the equal-weight
    ratio), i.e. one up to O(1/N); the statistic is recorded, with its
    dispersion across draws, as the check of that identity.
Next-direction diagnostic (out of sample): the realized variance ratio of the
    fixed portfolio along the (K+1)-th formation direction over the outcome
    sessions; one when the K-factor residuals carry no realized common variation
    along the strongest remaining formation direction.
"""
from __future__ import annotations
import numpy as np

from civol.core.common_space import standardize, gram_eigen, pca

BOUND = 1.25            # maximum residual variance ratio for rank selection
KMAX = 40               # rules search k = 0..KMAX; a rule that reaches KMAX is flagged
SPLIT = 500             # estimation sessions for rule B (validation = T - SPLIT = 250)
SIGN_DRAWS = 128        # balanced random sign vectors per date
SIGN_SEED = 2026102002  # base seed; the date index is added so draws differ by date and repeat by date


def portfolio_variance_ratio(E, w, scored=None):
    """Var(sum_i w_i E_i) / sum_i w_i^2 Var(E_i) over the sessions of E (rows), using
    realized (uncentered) second moments; one under uncorrelated columns. The
    equal-weight ratio is this with w = 1/N on centered E."""
    if scored is not None:
        E = E[:, scored]; w = w[scored]
    p = E @ w
    denominator = float(np.sum(w * w * np.sum(E * E, axis=0)))
    if denominator <= 0:
        return np.nan
    return float(np.sum(p * p) / denominator)


def equal_weight_ratio(E):
    """N Var(mean_i e_i) / mean_i Var(e_i) with column-centered E."""
    T, N = E.shape
    Ec = E - E.mean(0)
    p = Ec.mean(1)
    return float(N * np.mean(p * p) / np.mean(Ec * Ec))


def balanced_signs(N, draws, seed):
    """N x draws matrix of +-1 with exactly floor(N/2) minus signs per column."""
    rng = np.random.default_rng(int(seed))
    S = np.ones((N, draws))
    half = N // 2
    for r in range(draws):
        S[rng.permutation(N)[:half], r] = -1.
    return S


def random_signed_stats(E, S, equal_weight=None):
    """Variance ratios of the balanced random-signed portfolios E @ S / N relative
    to uncorrelated residuals: mean, sd, 5th/95th percentiles over draws, and the
    identity prediction (N - EW)/(N - 1) for the mean."""
    T, N = E.shape
    Ec = E - E.mean(0)
    P = Ec @ S / N
    ratios = N * np.mean(P * P, axis=0) / np.mean(Ec * Ec)
    ew = equal_weight_ratio(E) if equal_weight is None else equal_weight
    return dict(mean=float(ratios.mean()), sd=float(ratios.std(ddof=1)), p05=float(np.percentile(ratios, 5)), p95=float(np.percentile(ratios, 95)),
                identity_prediction=float((N - ew) / (N - 1)), draws=int(S.shape[1]))


def insample_curves(x, kmax=KMAX, signs=None, prepared=None):
    """Rule-A statistic and the random-signed check for k = 0..kmax on the full
    formation window. Residuals are the core's PCA residuals in return
    units (the degrees-of-freedom multiplier cancels in every ratio).
    prepared: optional (s, Z, (w, U)) from standardize/gram_eigen of the same x."""
    if prepared is None:
        mu, s, Z = standardize(x)
        w, U = gram_eigen(Z)
    else:
        s, Z, (w, U) = prepared
    T, N = Z.shape
    kmax = int(min(kmax, N - 2, T - 3))
    E = Z.copy()
    ew = np.empty(kmax + 1); rs_mean = np.empty(kmax + 1); rs_sd = np.empty(kmax + 1); rs_p95 = np.empty(kmax + 1)
    for k in range(kmax + 1):
        if k > 0:
            f = np.sqrt(T) * U[:, k - 1]
            b = Z.T @ f / T
            E -= np.outer(f, b)
        Er = E * s[None, :]
        ew[k] = equal_weight_ratio(Er)
        if signs is not None:
            st = random_signed_stats(Er, signs, ew[k])
            rs_mean[k], rs_sd[k], rs_p95[k] = st['mean'], st['sd'], st['p95']
    out = dict(equal_weight=ew.tolist(), eigenvalues=w[:kmax + 1].tolist(), T=T, N=N, kmax=kmax)
    if signs is not None:
        out.update(random_signed_mean=rs_mean.tolist(), random_signed_sd=rs_sd.tolist(), random_signed_p95=rs_p95.tolist())
    return out


def heldout_curve(x, split=SPLIT, kmax=KMAX):
    """Rule-B statistic for k = 0..kmax: the validation-window variance ratio of
    the (k+1)-th estimation-part principal direction relative to uncorrelated
    k-factor validation residuals. Validation residuals use the estimation-part
    loadings with factors extracted by cross-sectional least squares each
    session (the same construction as the out-of-sample residual proxy)."""
    x = np.asarray(x, float)
    T, N = x.shape
    if not 0 < split < T - 10:
        raise ValueError('split outside the formation window')
    # Exclude issuers with degenerate variance in the estimation window from
    # rule B: they cannot be standardized even if full-window variance is positive.
    x1 = x[:split]
    var1 = np.mean((x1 - x1.mean(0)) ** 2, axis=0)
    keep = var1 > 1e-14                                   # same variance threshold as standardize
    dropped = int(N - keep.sum())
    if dropped:
        x = x[:, keep]; N = int(keep.sum())
    mu1, s1, Z1 = standardize(x[:split])
    T1 = Z1.shape[0]
    w1, U1 = gram_eigen(Z1)
    kmax = int(min(kmax, N - 2, T1 - 3))
    B = Z1.T @ (np.sqrt(T1) * U1[:, :kmax + 1]) / T1          # N x (kmax+1) standardized loadings, orthogonal columns
    Z2 = (x[split:] - mu1) / s1
    ratios = np.empty(kmax + 1); numerators = np.empty(kmax + 1)
    E2 = Z2.copy()
    for k in range(kmax + 1):
        if k > 0:
            Bk = B[:, :k]
            F2 = np.linalg.solve(Bk.T @ Bk, Bk.T @ Z2.T).T         # cross-sectional least squares factors
            E2 = Z2 - F2 @ Bk.T
        v = B[:, k]
        ratios[k] = portfolio_variance_ratio(E2, v)
        numerators[k] = float(np.sum((E2 @ v) ** 2) / len(E2))
    return dict(ratio=ratios.tolist(), portfolio_variance=numerators.tolist(), estimation_eigenvalues=w1[:kmax + 1].tolist(),
                split=int(split), validation_sessions=int(T - split), N=N, kmax=kmax, dropped_degenerate=dropped)


def first_at_or_below(curve, bound):
    """Smallest index whose value is <= bound; (index, True) or (last index, False)."""
    c = np.asarray(curve, float)
    hits = np.flatnonzero(c <= bound)
    if len(hits):
        return int(hits[0]), True
    return int(len(c) - 1), False


def rule_a(insample, bound=BOUND):
    """Smallest k whose in-sample equal-weight ratio is at or below
    bound * max(1, floor), floor = min_k EW_k over k <= KMAX. When the floor
    exceeds one, selection uses the same proportional bound relative to that
    floor. Returns (k, attainable), where attainable indicates floor <= bound."""
    ew = np.asarray(insample['equal_weight'], float)
    floor = float(ew.min())
    reference = max(1., floor)
    k, _ = first_at_or_below(ew, bound * reference)
    return k, bool(floor <= bound)


def rule_b(heldout, bound=BOUND):
    """Smallest k whose held-out next-direction ratio is at or below bound;
    (KMAX, False) when no admissible k attains it."""
    return first_at_or_below(heldout['ratio'], bound)


def decomposition_at(Z, eig, s, k):
    """Full-window decomposition at a fixed rank, matching the
    common_space(...)['variant'] block: scores, standardized and return-unit
    loadings, degrees-of-freedom-corrected residuals in return units, scales."""
    T, N = Z.shape
    F, B, E = pca(Z, k, eig)
    E = E * np.sqrt(T / (T - k - 1)) * s[None, :]
    return dict(rank=int(k), F=F, B=B, loadings=s[:, None] * B, residuals=E, d0=np.mean(E * E, axis=0))


def next_direction(Z, eig, s, k):
    """Standardized loadings of the (k+1)-th full-window principal direction."""
    T, N = Z.shape
    w, U = eig
    return Z.T @ (np.sqrt(T) * U[:, k]) / T


def top_direction(E):
    """Standardized loadings of the first principal direction of a residual panel
    (used as the 'next direction' of the FF3 residual)."""
    Ec = E - E.mean(0)
    sd = np.sqrt(np.mean(Ec * Ec, axis=0)); sd[sd <= 0] = np.inf
    Es = Ec / sd
    w, U = gram_eigen(Es)
    T = len(Es)
    return Es.T @ (np.sqrt(T) * U[:, 0]) / T
