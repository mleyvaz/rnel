"""Existential (noisy-OR) aggregation of per-source stances, the glut, and calibration tools (EXPERIMENTAL).

Setting. A claim is examined by K >= 0 sources. Source k emits exactly one stance, drawn from a categorical
distribution (s_k, r_k, n_k) over {support, refute, insufficient}, with s_k + r_k + n_k = 1. Unless stated
otherwise, the stances of different sources are assumed *independent*. Under these assumptions the existential
triple of a claim is

    T = P(some source supports)  = 1 - prod_k (1 - s_k)
    F = P(some source refutes)   = 1 - prod_k (1 - r_k)
    I = P(every source is insufficient) = prod_k n_k

and the glut g = T + I + F - 1 is the probability that *some* source supports and *some* source refutes
(inclusion-exclusion). Reading each source as a Dempster-Shafer mass function on Theta = {s, r} with
m_k({s}) = s_k, m_k({r}) = r_k, m_k(Theta) = n_k, g also equals the conflict mass K of their conjunctive
(unnormalised Dempster) combination. The identity g = K holds for these product expressions as an algebraic identity; independence is what makes
T, F and g the probabilities of the named events. With dependent sources (copying, shared upstream evidence)
they are no longer those probabilities and should be recalibrated (see `mondrian_calibrate`).

The second half of the module contains distribution-free tools used on top of the triple:
  isotonic_fit / isotonic_predict   pool-adjacent-violators (PAVA) isotonic regression, no scikit-learn needed
  venn_abers                        inductive Venn-Abers predictor (Vovk and Petej 2014)
  mondrian_calibrate                isotonic calibration within strata (e.g. number of sources)
  glut_test                         split-conformal p-value for "this glut is unusually large"

All functions are vectorised with numpy; claims are indexed 0..n_claims-1 through an `owner` array that gives,
for every source row, the claim it belongs to.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = [
    "existential_triple",
    "glut",
    "belnap_masses",
    "in_faithful_region",
    "UnfaithfulTripleWarning",
    "dempster_conflict",
    "IsotonicModel",
    "isotonic_fit",
    "isotonic_predict",
    "venn_abers",
    "mondrian_calibrate",
    "glut_test",
]

_TOL = 1e-9


class UnfaithfulTripleWarning(UserWarning):
    """A (T, I, F) triple lies outside the region reachable from independent sources (some Belnap mass < 0)."""


# ---------------------------------------------------------------------------------------------------------------
# Existential triple
# ---------------------------------------------------------------------------------------------------------------

def _validate_profiles(s, r, n, owner, n_claims, tol=1e-8):
    s = np.asarray(s, dtype=float).ravel()
    r = np.asarray(r, dtype=float).ravel()
    n = np.asarray(n, dtype=float).ravel()
    owner = np.asarray(owner).ravel()
    if not (s.shape == r.shape == n.shape == owner.shape):
        raise ValueError("s, r, n and owner must have the same length (one entry per source)")
    if owner.size and (not np.issubdtype(owner.dtype, np.integer)):
        if not np.all(owner == np.floor(owner)):
            raise ValueError("owner must contain integer claim indices")
        owner = owner.astype(np.int64)
    owner = owner.astype(np.int64, copy=False)
    if owner.size and owner.min() < 0:
        raise ValueError("owner indices must be non-negative")
    if np.any(s < -tol) or np.any(r < -tol) or np.any(n < -tol):
        raise ValueError("stance probabilities must be non-negative")
    if np.any(np.abs(s + r + n - 1.0) > tol):
        raise ValueError("each source row (s, r, n) must sum to 1")
    s, r, n = np.clip(s, 0.0, 1.0), np.clip(r, 0.0, 1.0), np.clip(n, 0.0, 1.0)
    m = int(owner.max()) + 1 if owner.size else 0
    if n_claims is None:
        n_claims = m
    elif n_claims < m:
        raise ValueError(f"n_claims={n_claims} but owner refers to claim {m - 1}")
    return s, r, n, owner, int(n_claims)


def _log_prod(values, owner, n_claims):
    """Per-claim sum of log(values) (log of the product); empty claims give 0 (empty product = 1)."""
    with np.errstate(divide="ignore"):
        logs = np.log(values)
    out = np.zeros(n_claims)
    np.add.at(out, owner, logs)  # -inf propagates correctly (no +inf terms can occur)
    return out


def existential_triple(s, r, n, owner, n_claims=None):
    """Existential (noisy-OR) triple (T, I, F) of every claim from per-source stance probabilities.

    Parameters
    ----------
    s, r, n : array_like, shape (K,)
        Per-source probabilities of the stances support, refute and insufficient. Each row must be non-negative
        and sum to 1 (tolerance 1e-8): the model assumes each source emits exactly one stance.
    owner : array_like of int, shape (K,)
        Claim index of each source row.
    n_claims : int, optional
        Number of claims (default: max(owner) + 1). Claims without sources get T = F = 0, I = 1.

    Returns
    -------
    T, I, F : ndarray, shape (n_claims,)
        T = 1 - prod_k (1 - s_k), F = 1 - prod_k (1 - r_k), I = prod_k n_k, products taken over the sources of
        the claim. Products are computed in log space (log1p / expm1) for numerical stability with many sources.

    Notes
    -----
    Assumptions: one stance per source; stances of different sources independent. Then T, F and I are the
    probabilities that some source supports, some source refutes, and all sources are insufficient, and

        T + I + F - 1 = P(some source supports and some source refutes),

    which equals Dempster's conflict mass K = m_12...K(empty set) of the conjunctive combination of the mass
    functions m_k({s}) = s_k, m_k({r}) = r_k, m_k(Theta) = n_k (see `dempster_conflict`). With a single source
    the glut is zero (T + I + F = 1). T + I + F can exceed 1: the triple is paraconsistent ("glutty"), not a
    probability distribution over three exclusive outcomes. Without independence the formulas still define
    scores, but the probabilistic readings above are lost.
    """
    s, r, n, owner, m = _validate_profiles(s, r, n, owner, n_claims)
    with np.errstate(divide="ignore"):
        ls = np.log1p(-s)
        lr = np.log1p(-r)
    acc_s = np.zeros(m)
    acc_r = np.zeros(m)
    np.add.at(acc_s, owner, ls)
    np.add.at(acc_r, owner, lr)
    T = -np.expm1(acc_s)
    F = -np.expm1(acc_r)
    I = np.exp(_log_prod(n, owner, m))
    return T, I, F


def glut(T, I, F):
    """Glut g = T + I + F - 1 (elementwise).

    For a triple produced by `existential_triple` from independent sources, g = P(some support and some
    refutation) lies in [0, 1]. For arbitrary triples it may be negative (a "gap") or exceed what independent
    sources can produce; see `in_faithful_region`.
    """
    return np.asarray(T, dtype=float) + np.asarray(I, dtype=float) + np.asarray(F, dtype=float) - 1.0


def in_faithful_region(T, I, F, tol=_TOL):
    """Boolean mask: True where (t, f, b, n) of `belnap_masses` are all >= -tol and T, I, F lie in [-tol, 1+tol]."""
    T, I, F = (np.asarray(v, dtype=float) for v in (T, I, F))
    g = T + I + F - 1.0
    masses = (T - g, F - g, g, I)
    ok = np.ones(np.broadcast(T, I, F).shape, dtype=bool)
    for m in masses:
        ok &= m >= -tol
    for v in (T, I, F):
        ok &= (v >= -tol) & (v <= 1.0 + tol)
    return ok


def belnap_masses(T, I, F, strict=True, tol=_TOL):
    """Four-valued (Belnap-Dunn) masses (t, f, b, n) of a triple.

    t = T - g   (support only: some source supports, none refutes)
    f = F - g   (refutation only)
    b = g       (both: some source supports and some refutes)
    n = I       (neither: every source insufficient)
    with g = T + I + F - 1. They sum to 1 by construction.

    Assumption: the triple comes from independent sources (`existential_triple`); then t, f, b, n are the
    probabilities of the four exclusive events above and are all non-negative (the "faithful region").
    For other triples some masses may be negative. With strict=True (default) a ValueError is raised when any
    mass is below -tol; with strict=False an `UnfaithfulTripleWarning` is issued and the raw values are
    returned (they are then signed quantities, not probabilities). Values within tolerance are clipped to 0.
    """
    T, I, F = (np.asarray(v, dtype=float) for v in (T, I, F))
    g = T + I + F - 1.0
    t, f, b, nn = T - g, F - g, g, I + 0.0 * g
    ok = in_faithful_region(T, I, F, tol)
    if not np.all(ok):
        msg = f"{int(np.size(ok) - np.count_nonzero(ok))} triple(s) outside the faithful region (a Belnap mass < 0)"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UnfaithfulTripleWarning, stacklevel=2)
        return t, f, b, nn

    def _c(x):
        return np.where((x < 0) & (x >= -tol), 0.0, x)

    return _c(t), _c(f), _c(b), _c(nn)


def dempster_conflict(s, r, n, owner, n_claims=None):
    """Dempster conflict K of the conjunctive combination of the sources of each claim.

    Source k is the mass function on Theta = {s, r}: m_k({s}) = s_k, m_k({r}) = r_k, m_k(Theta) = n_k.
    Their unnormalised conjunctive combination gives non-empty focal sets only when all sources are in
    {s} or Theta (mass prod(s + n)), all in {r} or Theta (mass prod(r + n)), the overlap being all Theta
    (mass prod(n)); hence

        K = 1 - [prod_k (s_k + n_k) + prod_k (r_k + n_k) - prod_k n_k].

    This is computed directly (independently of `existential_triple`) and equals `glut` of the existential
    triple, which serves as a consistency check. Claims without sources have K = 0.
    """
    s, r, n, owner, m = _validate_profiles(s, r, n, owner, n_claims)
    ps = np.exp(_log_prod(s + n, owner, m))
    pr = np.exp(_log_prod(r + n, owner, m))
    pn = np.exp(_log_prod(n, owner, m))
    return 1.0 - (ps + pr - pn)


# ---------------------------------------------------------------------------------------------------------------
# Isotonic regression (PAVA)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class IsotonicModel:
    """Fitted non-decreasing step function: knots x (sorted, unique) and fitted values y in [0, 1]."""

    x: np.ndarray
    y: np.ndarray


def _pava(y, w):
    """Weighted pool-adjacent-violators for a non-decreasing fit of y (already ordered by x)."""
    k = y.size
    means = np.empty(k)
    weights = np.empty(k)
    sizes = np.empty(k, dtype=np.int64)
    top = -1
    for i in range(k):
        top += 1
        means[top], weights[top], sizes[top] = y[i], w[i], 1
        while top > 0 and means[top - 1] > means[top]:
            wt = weights[top - 1] + weights[top]
            means[top - 1] = (weights[top - 1] * means[top - 1] + weights[top] * means[top]) / wt
            weights[top - 1] = wt
            sizes[top - 1] += sizes[top]
            top -= 1
    return np.repeat(means[: top + 1], sizes[: top + 1])


def isotonic_fit(x, y, sample_weight=None):
    """Least-squares non-decreasing fit of y on x by the pool-adjacent-violators algorithm (PAVA).

    Ties in x are first pooled (weighted mean of their y), so the fit is a function of x. Fitted values are
    clipped to [0, 1] (the targets are meant to be probabilities or 0/1 labels). Implemented here because
    scikit-learn is not a dependency of rnel.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape or x.size == 0:
        raise ValueError("x and y must be non-empty and of equal length")
    w = np.ones_like(x) if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
    ux, inv = np.unique(x, return_inverse=True)
    sw = np.bincount(inv, weights=w, minlength=ux.size)
    sy = np.bincount(inv, weights=w * y, minlength=ux.size)
    fit = _pava(sy / sw, sw)
    return IsotonicModel(ux, np.clip(fit, 0.0, 1.0))


def isotonic_predict(model, x):
    """Evaluate a fitted isotonic model: linear interpolation between knots, constant beyond the end knots."""
    x = np.asarray(x, dtype=float)
    return np.clip(np.interp(x, model.x, model.y), 0.0, 1.0)


# ---------------------------------------------------------------------------------------------------------------
# Venn-Abers
# ---------------------------------------------------------------------------------------------------------------

def venn_abers(cal_scores, cal_labels, test_scores):
    """Inductive Venn-Abers predictor (Vovk and Petej 2014).

    For every test score x, isotonic regression is fitted twice to the calibration set extended with (x, 0)
    and with (x, 1); the fitted values at x give p0 and p1, the lower and upper probabilities that the label
    is 1. The merged point prediction is p = p1 / (1 - p0 + p1) (the log-loss-minimax merge).

    Assumptions: calibration and test pairs exchangeable; labels in {0, 1}; scores ordered so that larger
    means "more likely 1". Then (p0, p1) is a valid multiprobability prediction and p0 <= p1 always.

    Efficiency: the extended fit depends on x only through its position among the sorted calibration scores
    (the pair of left/right insertion ranks, which also records ties with existing scores). The two fits are
    computed once per distinct position and cached, so the cost is O(n * #distinct positions) rather than
    O(n * #test). The result is identical to refitting from scratch for each test point.

    Returns
    -------
    p0, p1, p : ndarray, shape (len(test_scores),)
    """
    cs = np.asarray(cal_scores, dtype=float).ravel()
    cl = np.asarray(cal_labels, dtype=float).ravel()
    ts = np.asarray(test_scores, dtype=float).ravel()
    if cs.shape != cl.shape:
        raise ValueError("cal_scores and cal_labels must have equal length")
    if not np.all((cl == 0) | (cl == 1)):
        raise ValueError("cal_labels must be 0/1")
    srt = np.sort(cs)
    left = np.searchsorted(srt, ts, side="left")
    right = np.searchsorted(srt, ts, side="right")
    p0 = np.empty(ts.size)
    p1 = np.empty(ts.size)
    cache: dict[tuple[int, int], tuple[float, float]] = {}
    xs = np.append(cs, 0.0)
    for j in range(ts.size):
        key = (int(left[j]), int(right[j]))
        hit = cache.get(key)
        if hit is None:
            x = ts[j]
            xs[-1] = x
            v0 = float(isotonic_predict(isotonic_fit(xs, np.append(cl, 0.0)), x))
            v1 = float(isotonic_predict(isotonic_fit(xs, np.append(cl, 1.0)), x))
            hit = cache[key] = (v0, v1)
        p0[j], p1[j] = hit
    p = p1 / (1.0 - p0 + p1)
    return p0, p1, p


# ---------------------------------------------------------------------------------------------------------------
# Mondrian calibration
# ---------------------------------------------------------------------------------------------------------------

def mondrian_calibrate(g_cal, y_cal, strata_cal, g_test, strata_test, min_size=40, return_info=False):
    """Isotonic calibration of a score separately within strata (Mondrian calibration).

    Typical use: the score is T (or g, or 1 - I) from `existential_triple`, the label says whether the event
    actually holds, and the stratum is the number of sources of the claim. Independence fails in practice
    (sources copy each other), and how badly depends on the number of sources, so each stratum gets its own
    monotone map from score to probability.

    Rules per test stratum h:
      * stratum seen in calibration with at least `min_size` points and both classes present: isotonic fit
        (PAVA) of y on g within h;
      * stratum seen in calibration but smaller than `min_size`, or containing a single class: the stratum
        base rate mean(y_cal | h) (a constant; too few points for a reliable monotone map);
      * stratum absent from calibration (both classes absent): global isotonic fit on all calibration data.

    Assumptions: within each stratum, calibration and test pairs are exchangeable and the probability of the
    label is non-decreasing in the score. Outputs are in [0, 1].

    Returns
    -------
    p : ndarray of calibrated probabilities for the test points; with return_info=True, (p, info) where
        info maps each test stratum to "isotonic", "base_rate" or "global".
    """
    g_cal = np.asarray(g_cal, dtype=float).ravel()
    y_cal = np.asarray(y_cal, dtype=float).ravel()
    strata_cal = np.asarray(strata_cal).ravel()
    g_test = np.asarray(g_test, dtype=float).ravel()
    strata_test = np.asarray(strata_test).ravel()
    if not (g_cal.shape == y_cal.shape == strata_cal.shape):
        raise ValueError("g_cal, y_cal and strata_cal must have equal length")
    if g_test.shape != strata_test.shape:
        raise ValueError("g_test and strata_test must have equal length")
    if g_cal.size == 0:
        raise ValueError("empty calibration set")
    global_model = None
    out = np.empty(g_test.size)
    info = {}
    for h in np.unique(strata_test):
        te = strata_test == h
        ca = strata_cal == h
        nh = int(ca.sum())
        if nh == 0:
            if global_model is None:
                global_model = isotonic_fit(g_cal, y_cal)
            out[te] = isotonic_predict(global_model, g_test[te])
            info[h.item() if hasattr(h, "item") else h] = "global"
            continue
        yh = y_cal[ca]
        both = (yh.min() < yh.max())
        if nh >= min_size and both:
            out[te] = isotonic_predict(isotonic_fit(g_cal[ca], yh), g_test[te])
            info[h.item() if hasattr(h, "item") else h] = "isotonic"
        else:
            out[te] = float(np.mean(yh))
            info[h.item() if hasattr(h, "item") else h] = "base_rate"
    out = np.clip(out, 0.0, 1.0)
    return (out, info) if return_info else out


# ---------------------------------------------------------------------------------------------------------------
# Conformal glut test
# ---------------------------------------------------------------------------------------------------------------

def glut_test(g_cal_nonconflict, g_test):
    """Split-conformal p-value for "the glut of this claim is unusually large".

    p(x) = (1 + #{i : g_cal_i >= g_test}) / (n + 1), with g_cal the gluts of n calibration claims known to be
    non-conflicting. Larger g is stranger, so small p is evidence of genuine conflict.

    Assumption: under the null (the test claim is non-conflicting) the n + 1 gluts are exchangeable. Then
    P(p <= alpha) <= alpha for every alpha in [0, 1] (exactly floor(alpha (n + 1)) / (n + 1) without ties).
    """
    cal = np.sort(np.asarray(g_cal_nonconflict, dtype=float).ravel())
    if cal.size == 0:
        raise ValueError("empty calibration set")
    gt = np.asarray(g_test, dtype=float)
    ge = cal.size - np.searchsorted(cal, gt, side="left")
    return (1.0 + ge) / (cal.size + 1.0)
