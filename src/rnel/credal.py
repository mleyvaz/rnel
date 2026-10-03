"""(Experimental) Credal ensembles, their neutrosophic reading, and conflict between views.

Requires numpy and scikit-learn (``pip install rnel[credal]``).

Three layers, kept separate on purpose:

1. **Credal ensemble** (:class:`CredalEnsemble`). A scikit-learn ensemble (a
   ``RandomForestClassifier`` by default) is read as a finite set of probability distributions, one
   per member. For every instance and class A it returns the lower and upper probabilities
   ``l_A = min_m p_m(A)`` and ``u_A = max_m p_m(A)`` over the members (``envelope="hull"``), or a
   more robust envelope (``"trimmed"``, ``"quantile"``). This is the credal-ensembling construction
   of V.-L. Nguyen, H. Zhang and S. Destercke, "Learning sets of probabilities through ensemble
   methods", in *Symbolic and Quantitative Approaches to Reasoning with Uncertainty (ECSQARU 2023)*,
   LNCS 14294, Springer, pp. 270-283, doi:10.1007/978-3-031-45608-4_21 (reference code:
   github.com/Haifei-ZHANG/Probability-Sets-Model, whose default forest uses 100 trees and
   ``min_samples_leaf=5``; fully grown trees give 0/1 leaf distributions and near-vacuous bounds).

2. **Neutrosophic reading** (:func:`to_neutrosophic`, :func:`from_neutrosophic`). T = l, I = u - l,
   F = 1 - u. This is a change of representation only and adds no information.

3. **What does add information**: evidence counts per member or per *view* (e.g. a time-domain and a
   frequency-domain model of the same signal), their fusion, the between-view conflict C* of
   ``rnel.conflict`` (binary one-vs-rest and its sup-norm multiclass extension), and the four states
   support / rejection / absence / conflict with the decision rules answer / abstain-and-seek-data /
   review-sources.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .sl import DEFAULT_W

__all__ = [
    "CredalEnsemble", "envelope", "coherent_bounds", "to_neutrosophic", "from_neutrosophic",
    "sup_gap_views", "c_star_multiclass", "c_star_one_vs_rest", "evidential_state", "decide",
    "ACTIONS",
]


# ----------------------------------------------------------------------------- envelopes
def envelope(P: np.ndarray, method: str = "hull", alpha: float = 0.0, q: float = 0.05):
    """Lower/upper probabilities of a set of member distributions.

    ``P`` has shape (M, n, K): M members, n instances, K classes (rows sum to 1).

    method = "hull":     l = min over members, u = max over members (the exact lower/upper
                         envelope of the finite credal set and of its convex hull).
    method = "trimmed":  discard, per instance, the fraction ``alpha`` of members farthest (Euclidean)
                         from the member mean, then take the hull of the rest (the alpha-trimming of
                         Nguyen, Zhang & Destercke 2023, 'SE' distance). Still the envelope of a set of
                         genuine distributions.
    method = "quantile": per class, the q and 1-q empirical quantiles of the member probabilities.
                         More robust to single outlying members, but the bounds of different classes
                         may come from different members, so the pair is not always the envelope of a
                         sub-population; pass it through :func:`coherent_bounds` if coherence matters.
    Returns (lower, upper), each of shape (n, K).
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 3:
        raise ValueError("P must have shape (members, instances, classes)")
    if method == "hull":
        return P.min(axis=0), P.max(axis=0)
    if method == "trimmed":
        if not 0 <= alpha < 1:
            raise ValueError("alpha must be in [0, 1)")
        M = P.shape[0]
        keep = M - int(np.floor(alpha * M))
        center = P.mean(axis=0, keepdims=True)
        d = np.sqrt(((P - center) ** 2).sum(axis=2))          # (M, n)
        idx = np.argsort(d, axis=0, kind="stable")[:keep]      # (keep, n)
        kept = np.take_along_axis(P, idx[:, :, None], axis=0)
        return kept.min(axis=0), kept.max(axis=0)
    if method == "quantile":
        if not 0 <= q < 0.5:
            raise ValueError("q must be in [0, 0.5)")
        return np.quantile(P, q, axis=0), np.quantile(P, 1 - q, axis=0)
    raise ValueError(f"unknown envelope {method!r}")


def coherent_bounds(lower: np.ndarray, upper: np.ndarray):
    """Tighten probability intervals to their reachable (coherent) version (de Campos et al. 1994):
    l_j' = max(l_j, 1 - sum_{k!=j} u_k), u_j' = min(u_j, 1 - sum_{k!=j} l_k).
    Hull and trimmed envelopes of genuine distributions are already reachable up to rounding."""
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    su, sl = upper.sum(axis=-1, keepdims=True), lower.sum(axis=-1, keepdims=True)
    lo = np.maximum(lower, 1 - (su - upper))
    up = np.minimum(upper, 1 - (sl - lower))
    return np.clip(lo, 0, 1), np.clip(up, 0, 1)


# ----------------------------------------------------------------------------- neutrosophic reading
def to_neutrosophic(lower, upper):
    """(T, I, F) = (l, u - l, 1 - u) for each class.

    This is ONLY a change of representation. By Theorem 1 of the companion paper on credal sets
    and neutrosophic triples, normalised neutrosophic triples (T, I, F >= 0, T + I + F = 1) are in
    one-to-one correspondence with lower/upper pairs 0 <= l <= u <= 1 (the inverse is
    :func:`from_neutrosophic`). Nothing is gained or lost: any decision, score or test computed from
    (T, I, F) can be computed from (l, u) and vice versa. In particular I = u - l is the credal width,
    not a new kind of indeterminacy. Information beyond the credal set must come from somewhere else:
    see :func:`evidential_state` (evidence counts per view and the between-view conflict C*).
    """
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    if np.any(lower < -1e-12) or np.any(upper > 1 + 1e-12) or np.any(lower > upper + 1e-12):
        raise ValueError("need 0 <= lower <= upper <= 1")
    return lower, upper - lower, 1 - upper


def from_neutrosophic(T, I, F):
    """Inverse of :func:`to_neutrosophic`: l = T, u = 1 - F (requires T + I + F = 1)."""
    T, I, F = (np.asarray(x, float) for x in (T, I, F))
    if np.any(np.abs(T + I + F - 1) > 1e-9) or min(T.min(initial=0), I.min(initial=0), F.min(initial=0)) < -1e-12:
        raise ValueError("need T, I, F >= 0 and T + I + F = 1")
    return T, 1 - F


# ----------------------------------------------------------------------------- the ensemble
class CredalEnsemble:
    """Credal wrapper around a fitted-member scikit-learn ensemble.

    Parameters
    ----------
    estimator : an unfitted scikit-learn ensemble exposing ``estimators_`` after ``fit``
        (RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, ...). Default:
        ``RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features="sqrt")``.
    envelope, alpha, q : see :func:`envelope`.
    random_state : used only to build the default forest.
    """

    def __init__(self, estimator=None, envelope: str = "hull", alpha: float = 0.0, q: float = 0.05,
                 random_state: Optional[int] = None):
        self.estimator = estimator
        self.envelope = envelope
        self.alpha = alpha
        self.q = q
        self.random_state = random_state

    def fit(self, X, y, **fit_params):
        est = self.estimator
        if est is None:
            from sklearn.ensemble import RandomForestClassifier
            est = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features="sqrt",
                                         random_state=self.random_state, n_jobs=-1)
        est.fit(X, y, **fit_params)
        self.estimator_ = est
        self.classes_ = est.classes_
        return self

    # members ---------------------------------------------------------------
    def _member_columns(self, member):
        """Column index in ``classes_`` of each class known to ``member``."""
        K = len(self.classes_)
        mc = np.asarray(getattr(member, "classes_", np.arange(K)))
        if np.issubdtype(mc.dtype, np.number) and np.array_equal(mc, np.arange(len(mc))):
            return mc.astype(int)  # members trained on encoded labels (forests, bagging)
        pos = {c: i for i, c in enumerate(self.classes_)}
        return np.array([pos[c] for c in mc])

    def member_proba(self, X) -> np.ndarray:
        """Member distributions, shape (M, n, K), columns aligned to ``classes_``."""
        members = self.estimator_.estimators_
        feats = getattr(self.estimator_, "estimators_features_", None)
        X = np.asarray(X)
        out = np.zeros((len(members), X.shape[0], len(self.classes_)))
        for m, mem in enumerate(members):
            Xm = X[:, feats[m]] if feats is not None else X
            out[m][:, self._member_columns(mem)] = mem.predict_proba(Xm)
        return out

    def predict_proba(self, X) -> np.ndarray:
        """Precise summary: mean member distribution (what the forest itself predicts)."""
        return self.member_proba(X).mean(axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def bounds(self, X):
        """(lower, upper), each (n, K)."""
        return envelope(self.member_proba(X), self.envelope, self.alpha, self.q)

    def width(self, X) -> np.ndarray:
        l, u = self.bounds(X)
        return u - l

    def neutrosophic(self, X):
        """(T, I, F) per class: :func:`to_neutrosophic` of :meth:`bounds` (representation only)."""
        return to_neutrosophic(*self.bounds(X))

    def evidence(self, X, mode: str = "leaf", scale: float = 1.0) -> np.ndarray:
        """Evidence counts per instance and class, shape (n, K).

        mode = "leaf":  for tree ensembles, the mean over trees of the training class counts in the
                        leaf reached by x (the counts behind the IDM leaf intervals of the cautious
                        random forest). Small totals = few training cases like x (absence).
        mode = "votes": number of members whose most probable class is each class (total = M).
        mode = "mean":  ``scale`` times the mean member distribution (total = scale).
        """
        if mode == "votes":
            P = self.member_proba(X)
            K = P.shape[2]
            return np.stack([(P.argmax(axis=2) == k).sum(axis=0) for k in range(K)], axis=1).astype(float)
        if mode == "mean":
            return scale * self.predict_proba(X)
        if mode == "leaf":
            X = np.asarray(X)
            members = self.estimator_.estimators_
            feats = getattr(self.estimator_, "estimators_features_", None)
            out = np.zeros((X.shape[0], len(self.classes_)))
            for m, tree in enumerate(members):
                if not hasattr(tree, "tree_"):
                    raise ValueError("mode='leaf' needs tree members")
                Xm = X[:, feats[m]] if feats is not None else X
                leaves = tree.apply(Xm.astype(np.float32))
                v = tree.tree_.value[leaves, 0, :]                     # (n, K_member)
                # scikit-learn >= 1.4 stores class fractions in tree_.value; older versions store counts
                if np.allclose(v.sum(axis=1), 1.0):
                    counts = v * tree.tree_.weighted_n_node_samples[leaves][:, None]
                else:
                    counts = v
                out[:, self._member_columns(tree)] += counts
            return scale * out / len(members)
        raise ValueError(f"unknown evidence mode {mode!r}")


# ----------------------------------------------------------------------------- conflict between views
def _views(E) -> np.ndarray:
    E = np.asarray(E, dtype=float)
    if E.ndim == 2:
        E = E[:, None, :]
    if E.ndim != 3:
        raise ValueError("E must have shape (views, instances, classes)")
    if np.any(E < 0):
        raise ValueError("evidence counts must be non-negative")
    return E


def sup_gap_views(E) -> np.ndarray:
    """Multiclass between-source conflict per instance: sum_v max_k e_vk - max_k sum_v e_vk.

    Vectorised form of ``rnel.conflict.sup_gap``; for K = 2 it equals K_b = min(M+, M-)."""
    E = _views(E)
    return E.max(axis=2).sum(axis=0) - E.sum(axis=0).max(axis=1)


def c_star_multiclass(E, W: float = DEFAULT_W) -> np.ndarray:
    """Sup-norm multiclass C* = 2 * sup_gap / (N + W), N = total evidence of all views.

    Reduces to the binary C* of ``rnel.conflict`` when K = 2. For two views it lies in [0, 1); for
    V views its maximum is 2 (V - 1) / V. Invariant to the order of the views (sums and maxima)."""
    E = _views(E)
    return 2 * sup_gap_views(E) / (E.sum(axis=(0, 2)) + W)


def _binary_counts(E, cls_idx):
    E = _views(E)
    cls_idx = np.broadcast_to(np.asarray(cls_idx, int), (E.shape[1],))
    r = np.take_along_axis(E, cls_idx[None, :, None], axis=2)[..., 0]  # (V, n)
    s = E.sum(axis=2) - r
    return r, s


def c_star_one_vs_rest(E, cls_idx, W: float = DEFAULT_W) -> np.ndarray:
    """Binary C* (paper 3) about the proposition 'class = cls_idx', each view reporting
    (r_v, s_v) = (evidence for that class, evidence for the others)."""
    return evidential_state(E, cls_idx, W)["c"]


def evidential_state(E, cls_idx, W: float = DEFAULT_W) -> dict:
    """Support / rejection / absence / conflict for 'class = cls_idx' from per-view evidence.

    With R, S the pooled counts for and against, K_w = sum_v min(r_v, s_v), K_b = min(R, S) - K_w and
    tot = R + S + W:
        a = (R - K_b) / tot   support not in dispute between views
        b = (S - K_b) / tot   rejection not in dispute between views
        u = W / tot           absence of evidence
        c = 2 K_b / tot       conflict between views (= C* of rnel.conflict)
    so a + b + u + c = 1, and the paraconsistent reading is T = a + c, F = b + c, I = u + c
    (T + I + F = 1 + 2c). The state depends on the views only through the additive (R, S, K_w), so
    it does not depend on their order."""
    r, s = _binary_counts(E, cls_idx)
    R, S = r.sum(axis=0), s.sum(axis=0)
    Kw = np.minimum(r, s).sum(axis=0)
    Kb = np.maximum(0.0, np.minimum(R, S) - Kw)
    tot = R + S + W
    a, b, u, c = (R - Kb) / tot, (S - Kb) / tot, W / tot, 2 * Kb / tot
    return {"a": a, "b": b, "u": u, "c": c, "T": a + c, "F": b + c, "I": u + c,
            "R": R, "S": S, "Kw": Kw, "Kb": Kb}


ACTIONS = {
    "answer": "answer with the class (support dominates)",
    "abstain_seek_data": "abstain and collect more data (absence of evidence)",
    "review_sources": "abstain and review the sources/views (conflict between them)",
    "reject": "the class is rejected by the evidence; do not answer with it",
}


def decide(state: dict, tau_conflict: float = 0.2, tau_absence: float = 0.3,
           tau_support: float = 0.5) -> np.ndarray:
    """Action per instance, checked in this order: conflict (c >= tau_conflict) -> review_sources;
    absence (u >= tau_absence) -> abstain_seek_data; support (a >= tau_support) -> answer;
    otherwise reject. Thresholds are the user's; choose them on validation data, not on test."""
    c, u, a = (np.atleast_1d(state[k]) for k in ("c", "u", "a"))
    out = np.full(c.shape, "reject", dtype=object)
    out[a >= tau_support] = "answer"
    out[u >= tau_absence] = "abstain_seek_data"
    out[c >= tau_conflict] = "review_sources"
    return out
