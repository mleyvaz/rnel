"""(Experimental) Credal tools for neutrosophic triples.

Requires numpy and scipy (``pip install rnel[credal]``); the convenience function
:func:`credal_diagnosis` also accepts a fitted :class:`rnel.credal.CredalEnsemble`.

A neutrosophic probability assigns to an event A a triple NP(A) = (T, I, F). Under the betting reading
of Walley (1991), T(A) is an acceptable buying price for the gamble 1_A (a lower probability of A) and
F(A) one for 1_{not A}, so the triple determines the credal set

    M(NP) = { P : T(A) <= P(A) <= 1 - F(A) for every assessed A }.

Through this reading the neutrosophic triple inherits Walley's quality checks and tools, which this
module applies directly to triples:

* avoiding sure loss and its degree (Theorem 3), coherence (Theorems 1 and 2);
* the natural extension to any composite event, returned again as a triple, and the coherent
  correction of an assessment, which never increases indeterminacy (Theorem 1(c));
* singleton assessments as probability intervals, with closed forms (Theorem 2) and the imprecise
  Dirichlet model (Corollary 2);
* decisions by sets: interval dominance, maximality and E-admissibility;
* the glut lifting (Theorem 8), which represents the whole cube [0, 1]^3, including T + F > 1, by a
  non-empty credal set on a frame with a glut atom, and the N-norms as natural extensions on it
  (Theorem 9); the retraction to Subjective Logic (Theorem 5).

Theorem numbers refer to M. Y. Leyva-Vázquez and F. Smarandache, *Neutrosophic probability and credal
sets: exact reduction theorems and a glut lifting* (manuscript v0.5, 2026).

Conventions. The frame is a finite list of outcomes (labels or ``range(n)``). An assessment is either a
mapping ``{event: (T, I, F)}`` where an event is an outcome label or an iterable of labels, or a sequence
of K triples, read as the singletons of the K outcomes. Under the betting reading the upper probability
is 1 - F; when T + I + F = 1 this equals T + I and I = U - L is the width of the interval (Theorem 1).
The betting reading does not use I (Theorem 3(c)); I is recovered as 1 - T - F after a natural extension.
All linear programs use scipy's HiGHS solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.optimize import linprog

from .sl import DEFAULT_W

__all__ = [
    "Frame", "avoids_sure_loss", "sure_loss_degree", "is_coherent", "classify", "natural_extension",
    "coherent_correction", "singleton_intervals_check", "singleton_natural_extension",
    "singleton_correction", "idm_triple", "interval_dominance", "maximality", "e_admissible",
    "GlutLifting", "to_glut_frame", "glut_conjunction", "sl_retraction", "credal_diagnosis",
]

TOL = 1e-9
_LP_OPTS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}

Triple = tuple[float, float, float]


# ============================================================================ frame and assessments
@dataclass(frozen=True)
class Frame:
    """A finite frame of outcomes. ``mask(event)`` gives the 0/1 indicator of an event."""

    outcomes: tuple

    @classmethod
    def of(cls, outcomes) -> "Frame":
        if isinstance(outcomes, Frame):
            return outcomes
        if isinstance(outcomes, (int, np.integer)):
            return cls(tuple(range(int(outcomes))))
        return cls(tuple(outcomes))

    @property
    def n(self) -> int:
        return len(self.outcomes)

    def index(self, label) -> int:
        try:
            return self.outcomes.index(label)
        except ValueError:
            raise KeyError(f"{label!r} is not an outcome of the frame {self.outcomes}") from None

    def members(self, event) -> frozenset:
        """Indices of the outcomes in ``event`` (a label or an iterable of labels)."""
        if _is_label(event, self):
            return frozenset({self.index(event)})
        return frozenset(self.index(x) for x in event)

    def mask(self, event) -> np.ndarray:
        v = np.zeros(self.n)
        v[list(self.members(event))] = 1.0
        return v


def _is_label(x, frame: Optional[Frame] = None) -> bool:
    if frame is not None and x in frame.outcomes:
        return True
    return isinstance(x, (str, bytes, int, np.integer)) or not isinstance(x, Iterable)


def _is_triple_sequence(obj) -> bool:
    if isinstance(obj, Mapping):
        return False
    arr = np.asarray(obj, dtype=float)
    return arr.ndim == 2 and arr.shape[1] == 3


def _assessment(np_events, outcomes=None):
    """Return (frame, list of (mask, T, I, F, key)) from a mapping or a sequence of singleton triples."""
    if _is_triple_sequence(np_events):
        arr = np.asarray(np_events, dtype=float)
        frame = Frame.of(outcomes if outcomes is not None else arr.shape[0])
        if frame.n != arr.shape[0]:
            raise ValueError("one triple per outcome is needed")
        rows = []
        for i, (T, I, F) in enumerate(arr):
            m = np.zeros(frame.n)
            m[i] = 1.0
            rows.append((m, float(T), float(I), float(F), frame.outcomes[i]))
        return frame, rows
    if not isinstance(np_events, Mapping):
        raise TypeError("np_events must be a mapping {event: (T, I, F)} or a sequence of (T, I, F)")
    if outcomes is None:
        seen = []
        for ev in np_events:
            for x in ([ev] if _is_label(ev) else ev):
                if x not in seen:
                    seen.append(x)
        outcomes = seen
    frame = Frame.of(outcomes)
    rows = []
    for ev, trip in np_events.items():
        T, I, F = (float(v) for v in trip)
        rows.append((frame.mask(ev), T, I, F, ev))
    return frame, rows


def _singletons_only(frame: Frame, rows) -> Optional[np.ndarray]:
    """(K, 3) array if the assessment is exactly one triple per singleton, else None."""
    if len(rows) != frame.n:
        return None
    arr = np.full((frame.n, 3), np.nan)
    for m, T, I, F, _ in rows:
        if m.sum() != 1:
            return None
        i = int(np.argmax(m))
        if not np.isnan(arr[i, 0]):
            return None
        arr[i] = (T, I, F)
    return arr


# ============================================================================ linear programs
def _constraints(rows, n, eps=False):
    """Rows of P(A) >= T (- eps) and P(A) <= 1 - F (+ eps) in A_ub x <= b_ub form."""
    A, b = [], []
    for m, T, _I, F, _ in rows:
        if eps:
            A.append(np.r_[-m, -1.0]); b.append(-T)
            A.append(np.r_[m, -1.0]); b.append(1.0 - F)
        else:
            A.append(-m); b.append(-T)
            A.append(m); b.append(1.0 - F)
    if not A:
        return None, None
    return np.array(A), np.array(b)


def _lp(c, A_ub, b_ub, n, extra_vars=0):
    A_eq = np.r_[np.ones(n), np.zeros(extra_vars)].reshape(1, -1)
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                   bounds=[(0, None)] * (n + extra_vars), method="highs", options=_LP_OPTS)


def _lp_extreme(rows, n, mask, upper=False) -> Optional[float]:
    """min (or max) of P(event) over M(NP); None if M(NP) is empty."""
    A_ub, b_ub = _constraints(rows, n)
    c = mask * (-1.0 if upper else 1.0)
    r = _lp(c, A_ub, b_ub, n)
    if r.status != 0:
        return None
    return float(-r.fun if upper else r.fun)


def _lp_sure_loss(rows, n) -> float:
    A_ub, b_ub = _constraints(rows, n, eps=True)
    if A_ub is None:
        return 0.0
    r = _lp(np.r_[np.zeros(n), 1.0], A_ub, b_ub, n, extra_vars=1)
    if r.status != 0:  # pragma: no cover - the relaxed problem is always feasible
        raise RuntimeError(r.message)
    return max(0.0, float(r.fun))


# ============================================================================ sure loss (Theorem 3)
def _as_single_triple(x) -> Optional[Triple]:
    if isinstance(x, Mapping):
        return None
    arr = np.asarray(x, dtype=float)
    if arr.shape == (3,):
        return float(arr[0]), float(arr[1]), float(arr[2])
    return None


def sure_loss_degree(np_events, outcomes=None) -> float:
    """Degree of sure loss: the least uniform relaxation eps >= 0 of all bounds that makes
    {P : T(A) - eps <= P(A) <= 1 - F(A) + eps} non-empty. It is 0 exactly when sure loss is avoided.

    For a single triple ``(T, I, F)`` of one event on the frame {A, not A}, Theorem 3 gives the closed
    form (T + F - 1)^+ / 2 when T, F are in [0, 1]; with off values it is
    max((T + F - 1) / 2, T - 1, F - 1, 0). I plays no role (Theorem 3(c)): under the betting reading the
    zone T + F > 1 is exactly the zone of sure loss. For a general assessment it is a linear program.
    """
    single = _as_single_triple(np_events)
    if single is not None:
        T, _I, F = single
        return max((T + F - 1) / 2, T - 1, F - 1, 0.0)
    frame, rows = _assessment(np_events, outcomes)
    return _lp_sure_loss(rows, frame.n)


def avoids_sure_loss(np_events, outcomes=None, tol: float = TOL) -> bool:
    """True when the credal set M(NP) is non-empty (Walley's avoiding sure loss; Theorem 1(a)).

    For a single triple this is T + F <= 1 (Theorem 3(a)); for singleton triples it is
    0 <= Delta <= sum_i (1 - T_i - F_i) with Delta = 1 - sum_i T_i (Theorem 2(a))."""
    single = _as_single_triple(np_events)
    if single is not None:
        return sure_loss_degree(single) <= tol
    frame, rows = _assessment(np_events, outcomes)
    arr = _singletons_only(frame, rows)
    if arr is not None:
        return singleton_intervals_check(arr, tol=tol)["avoids_sure_loss"]
    return _lp_sure_loss(rows, frame.n) <= tol


# ============================================================================ coherence (Theorems 1, 2)
def is_coherent(np_events, outcomes=None, method: str = "auto", tol: float = 1e-7) -> bool:
    """Walley coherence: M(NP) is non-empty and every bound T(A), 1 - F(A) is attained on it.

    By Theorem 1(b) this is exactly the coherence of the lower/upper pair (T, 1 - F); it is a condition
    in addition to normalisation T + I + F = 1 (Theorem 1(d)). ``method="closed"`` (or ``"auto"`` for
    one triple per singleton with components in [0, 1]) uses Theorem 2(b): with Delta = 1 - sum_i T_i,
    coherent iff I_i <= Delta <= sum_{j != i} I_j for all i (I_i = 1 - T_i - F_i, the interval width).
    ``method="lp"`` solves two linear programs per assessed event."""
    if method not in ("auto", "lp", "closed"):
        raise ValueError("method must be 'auto', 'lp' or 'closed'")
    single = _as_single_triple(np_events)
    if single is not None:
        T, _I, F = single
        return 0 <= T <= 1 and 0 <= F <= 1 and T + F <= 1 + tol
    frame, rows = _assessment(np_events, outcomes)
    arr = _singletons_only(frame, rows)
    if method != "lp" and arr is not None and np.all(arr[:, [0, 2]] >= -tol) and np.all(arr[:, [0, 2]] <= 1 + tol):
        return singleton_intervals_check(arr, tol=tol)["coherent"]
    if method == "closed":
        raise ValueError("the closed form (Theorem 2) needs one triple per singleton with T, F in [0, 1]")
    n = frame.n
    if _lp_sure_loss(rows, n) > tol:
        return False
    for m, T, _I, F, _ in rows:
        lo = _lp_extreme(rows, n, m)
        hi = _lp_extreme(rows, n, m, upper=True)
        if abs(lo - T) > tol or abs(hi - (1 - F)) > tol:
            return False
    return True


def classify(np_events, outcomes=None) -> str:
    """'sure_loss', 'incoherent' (avoids sure loss, some bound not attained) or 'coherent'."""
    if not avoids_sure_loss(np_events, outcomes):
        return "sure_loss"
    return "coherent" if is_coherent(np_events, outcomes) else "incoherent"


# ============================================================================ natural extension
def _triple_from_bounds(lo: float, hi: float) -> Triple:
    return lo, hi - lo, 1.0 - hi


def natural_extension(np_triples, event, outcomes=None, method: str = "auto") -> Triple:
    """Neutrosophic natural extension NP^E(E) = (L(E), U(E) - L(E), 1 - U(E)) of an event E, where
    L and U are the lower and upper envelopes of M(NP) (Theorem 1(c)). E may be a composite event such as
    ``("A", "C")`` ("A or C"). The result is normalised and coherent.

    ``method="auto"`` uses the closed form of Theorem 2(c) when the assessment is one triple per singleton,
    and a linear program otherwise; ``"lp"`` forces the linear program; ``"closed"`` forces Theorem 2.
    Raises ValueError when the assessment incurs sure loss (the credal set is empty)."""
    frame, rows = _assessment(np_triples, outcomes)
    arr = _singletons_only(frame, rows)
    if method == "closed" or (method == "auto" and arr is not None):
        if arr is None:
            raise ValueError("the closed form (Theorem 2) needs one triple per singleton")
        return singleton_natural_extension(arr, frame.members(event), outcomes=None)
    if method not in ("auto", "lp"):
        raise ValueError("method must be 'auto', 'lp' or 'closed'")
    mask = frame.mask(event)
    lo = _lp_extreme(rows, frame.n, mask)
    if lo is None:
        raise ValueError(f"the assessment incurs sure loss (degree {_lp_sure_loss(rows, frame.n):.6g}); "
                         "no natural extension exists")
    hi = _lp_extreme(rows, frame.n, mask, upper=True)
    return _triple_from_bounds(lo, hi)


def coherent_correction(np_events, outcomes=None, method: str = "auto"):
    """Replace every assessed triple by its natural extension (Theorem 1(c)).

    The corrected assessment is coherent and has the same credal set. T and F can only increase, and
    I^E = 1 - T^E - F^E <= 1 - T - F; for normalised triples (T + I + F = 1) this is I^E <= I:
    correcting an incoherent assessment never adds indeterminacy. Returns the same container type as the
    input (a dict for a mapping, a (K, 3) array for singleton triples). Raises ValueError on sure loss."""
    frame, rows = _assessment(np_events, outcomes)
    arr = _singletons_only(frame, rows)
    if _is_triple_sequence(np_events):
        if method == "lp":
            return np.array([natural_extension(np_events, [o], frame.outcomes, method="lp")
                             for o in frame.outcomes])
        return singleton_correction(arr)
    return {key: natural_extension(np_events, key, frame.outcomes, method="lp" if arr is None else method)
            for (_m, _T, _I, _F, key) in rows}


# ============================================================================ singletons (Theorem 2)
def _singleton_bounds(triples):
    arr = np.asarray(triples, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("triples must have shape (K, 3)")
    return arr[:, 0], 1.0 - arr[:, 2]


def singleton_intervals_check(triples, tol: float = TOL) -> dict:
    """Theorem 2 on one triple per outcome, read as probability intervals l_i = T_i, u_i = 1 - F_i
    (de Campos, Huete and Moral 1994). With I_i = 1 - T_i - F_i (= the assessed I when normalised) and
    the truth deficit Delta = 1 - sum_i T_i:

    (a) avoids sure loss   iff 0 <= Delta <= sum_i I_i (and every I_i >= 0);
    (b) coherent           iff I_i <= Delta <= sum_{j != i} I_j for every i.

    Returns a dict with ``delta``, ``I`` (widths), ``normalised``, ``avoids_sure_loss``, ``coherent`` and
    the outcomes whose own indeterminacy exceeds the deficit (``I_above_delta``) or whose deficit cannot
    be covered by the others (``delta_above_rest``)."""
    arr = np.asarray(triples, dtype=float)
    l, u = _singleton_bounds(arr)
    I = u - l
    delta = 1.0 - l.sum()
    in_range = bool(np.all(l >= -tol) and np.all(u <= 1 + tol))
    # avoiding sure loss does not change when l, u are clipped to [0, 1] (same credal set)
    lc, uc = np.clip(l, 0, 1), np.clip(u, 0, 1)
    dc = 1.0 - lc.sum()
    asl = bool(np.all(uc - lc >= -tol) and dc >= -tol and dc <= (uc - lc).sum() + tol)
    above = [i for i in range(len(l)) if I[i] > delta + tol]
    rest = [i for i in range(len(l)) if delta > I.sum() - I[i] + tol]
    coherent = asl and in_range and not above and not rest
    return {"delta": float(delta), "I": I, "normalised": bool(np.allclose(arr.sum(axis=1), 1.0)),
            "avoids_sure_loss": asl, "coherent": bool(coherent),
            "I_above_delta": above, "delta_above_rest": rest}


def singleton_correction(triples) -> np.ndarray:
    """Natural extension of each singleton (reachable intervals of de Campos et al. 1994) in triple form:
    l_i' = max(l_i, 1 - sum_{j != i} u_j), u_i' = min(u_i, 1 - sum_{j != i} l_j), with l = T, u = 1 - F,
    and l, u first clipped to [0, 1]. The result satisfies Theorem 2(b), and I_i' <= I_i (Theorem 1(c)).
    Raises ValueError on sure loss."""
    arr = np.asarray(triples, dtype=float)
    l, u = _singleton_bounds(arr)
    l, u = np.clip(l, 0, 1), np.clip(u, 0, 1)
    if not singleton_intervals_check(np.c_[l, u - l, 1 - u])["avoids_sure_loss"]:
        raise ValueError(f"the assessment incurs sure loss (degree {sure_loss_degree(arr):.6g})")
    su, sl = u.sum(), l.sum()
    l2 = np.maximum(l, 1 - (su - u))
    u2 = np.minimum(u, 1 - (sl - l))
    return np.c_[l2, u2 - l2, 1 - u2]


def singleton_natural_extension(triples, event, outcomes=None) -> Triple:
    """Closed-form natural extension of Theorem 2(c) for one triple per outcome.

    After the singleton correction (a no-op when the assessment is coherent), with T_A = sum_{i in A} T_i,
    I_A = sum_{i in A} I_i, Delta = 1 - sum_i T_i:

        L(A) = T_A + (Delta - I_{not A})^+,   U(A) = T_A + min(I_A, Delta),
        NP^E(A) = (L(A), U(A) - L(A), 1 - U(A)).

    ``event`` is an iterable of outcome indices, or of labels when ``outcomes`` is given."""
    arr = singleton_correction(triples)
    K = arr.shape[0]
    frame = Frame.of(outcomes if outcomes is not None else K)
    idx = frame.members(event)
    inA = np.zeros(K, dtype=bool)
    inA[list(idx)] = True
    T, I = arr[:, 0], arr[:, 1]
    delta = 1.0 - T.sum()
    if not inA.any():
        return 0.0, 0.0, 1.0
    if inA.all():
        return 1.0, 0.0, 0.0
    TA, IA, InotA = T[inA].sum(), I[inA].sum(), I[~inA].sum()
    lo = TA + max(delta - InotA, 0.0)
    hi = TA + min(IA, delta)
    return _triple_from_bounds(float(lo), float(hi))


def idm_triple(counts, s: float = 2.0, event=None, outcomes=None):
    """Imprecise Dirichlet model (Walley 1996) as neutrosophic triples (Corollary 2).

    With counts n_i, N = sum_i n_i and prior strength s, the evidential triple of outcome i is
    (n_i / (N + s), s / (N + s), (N - n_i) / (N + s)); it is coherent, and its natural extension to any
    event A (neither empty nor the whole frame) is the IDM interval [n_A / (N + s), (n_A + s) / (N + s)].
    Returns a (K, 3) array, or the triple of ``event`` when given. For K = 2 and s = W this is the
    Subjective Logic opinion from evidence (Theorem 5(b))."""
    c = np.asarray(counts, dtype=float)
    if c.ndim != 1 or np.any(c < 0) or s <= 0:
        raise ValueError("counts must be a non-negative vector and s > 0")
    N = c.sum()
    if event is None:
        return np.c_[c / (N + s), np.full(len(c), s / (N + s)), (N - c) / (N + s)]
    frame = Frame.of(outcomes if outcomes is not None else len(c))
    idx = list(frame.members(event))
    if not idx:
        return 0.0, 0.0, 1.0
    if len(idx) == len(c):
        return 1.0, 0.0, 0.0
    nA = c[idx].sum()
    return float(nA / (N + s)), float(s / (N + s)), float((N - nA) / (N + s))


# ============================================================================ decisions by sets
def _decision_input(triples, members):
    """Return (K, kind, data) with kind 'intervals' (l, u reachable) or 'hull' (member matrix)."""
    if (triples is None) == (members is None):
        raise ValueError("give exactly one of triples or members")
    if members is not None:
        P = np.asarray(members, dtype=float)
        if P.ndim != 2 or np.any(np.abs(P.sum(axis=1) - 1) > 1e-6) or np.any(P < -1e-12):
            raise ValueError("members must have shape (M, K) with rows on the simplex")
        return P.shape[1], "hull", P
    arr = singleton_correction(triples)
    return arr.shape[0], "intervals", (arr[:, 0], 1 - arr[:, 2])


def _labels(idx, labels):
    return [labels[i] for i in idx] if labels is not None else list(idx)


def interval_dominance(triples=None, *, members=None, labels=None) -> list:
    """Classes not interval-dominated: k is discarded when some j has L(j) > U(k).

    ``triples`` (K, 3) define the credal set {p : T_k <= p_k <= 1 - F_k}; the bounds used are its
    natural extension (singleton correction). ``members`` (M, K) instead take the convex hull of a finite
    set of distributions, whose bounds are the member minima and maxima. Returns the non-dominated classes
    (labels when given, else indices). Interval dominance is the least decisive of the three rules: its
    set contains the maximal set, which contains the E-admissible set."""
    K, kind, data = _decision_input(triples, members)
    l, u = data if kind == "intervals" else (data.min(axis=0), data.max(axis=0))
    keep = [k for k in range(K) if not any(l[j] > u[k] + TOL for j in range(K) if j != k)]
    return _labels(keep, labels)


def _min_difference(kind, data, j, k) -> float:
    """min over the credal set of p_j - p_k."""
    if kind == "hull":
        return float((data[:, j] - data[:, k]).min())
    l, u = data
    K = len(l)
    c = np.zeros(K)
    c[j], c[k] = 1.0, -1.0
    r = linprog(c, A_eq=np.ones((1, K)), b_eq=[1.0], bounds=list(zip(l, u)), method="highs",
                options=_LP_OPTS)
    return float(r.fun)


def maximality(triples=None, *, members=None, labels=None) -> list:
    """Walley's maximality for the 0-1 utility: k is discarded when some j satisfies
    min_{p in M} (p_j - p_k) > 0, i.e. j is strictly preferred to k by every distribution of the set.
    Arguments as in :func:`interval_dominance`."""
    K, kind, data = _decision_input(triples, members)
    keep = [k for k in range(K)
            if not any(_min_difference(kind, data, j, k) > TOL for j in range(K) if j != k)]
    return _labels(keep, labels)


def e_admissible(triples=None, *, members=None, labels=None) -> list:
    """E-admissibility (Levi) for the 0-1 utility: k is kept when some p in the credal set has
    p_k >= p_j for every j (k is a Bayes act for some member of the set). One feasibility LP per class.
    Arguments as in :func:`interval_dominance`."""
    K, kind, data = _decision_input(triples, members)
    keep = []
    for k in range(K):
        if kind == "hull":
            M = data.shape[0]
            # weights lambda on the members; p = lambda @ data; constraints p_j - p_k <= 0
            A = np.array([data[:, j] - data[:, k] for j in range(K) if j != k])
            r = linprog(np.zeros(M), A_ub=A if len(A) else None, b_ub=np.zeros(len(A)) if len(A) else None,
                        A_eq=np.ones((1, M)), b_eq=[1.0], bounds=[(0, None)] * M, method="highs",
                        options=_LP_OPTS)
        else:
            l, u = data
            A = []
            for j in range(K):
                if j != k:
                    row = np.zeros(K)
                    row[j], row[k] = 1.0, -1.0
                    A.append(row)
            A = np.array(A)
            r = linprog(np.zeros(K), A_ub=A if len(A) else None, b_ub=np.zeros(len(A)) if len(A) else None,
                        A_eq=np.ones((1, K)), b_eq=[1.0], bounds=list(zip(l, u)), method="highs",
                        options=_LP_OPTS)
        if r.status == 0:
            keep.append(k)
    return _labels(keep, labels)


# ============================================================================ glut lifting (Theorems 8, 9)
_FRAMES = {
    # patterns (in E_T, in E_I, in E_F) of each atom
    "minimal": (("tf_glut_i_gap", (0, 1, 1)), ("glut", (1, 0, 1)), ("tf_gap", (1, 1, 0)), ("all", (1, 1, 1))),
    "belnap": (("t", (1, 0, 0)), ("f", (0, 0, 1)), ("b", (1, 0, 1)), ("n", (0, 1, 0))),
    "disjoint": (("a", (1, 0, 0)), ("iota", (0, 1, 0)), ("not_a", (0, 0, 1))),
}


@dataclass
class GlutLifting:
    """Credal set K(T, I, F) = {P : P(E_T) >= T, P(E_I) >= I, P(E_F) >= F} on a frame of atoms
    (Theorem 8). ``patterns[a]`` says whether atom a lies in (E_T, E_I, E_F).

    frame = "minimal": the four atoms with patterns (0,1,1), (1,0,1), (1,1,0), (1,1,1). Faithful and
    coherent on the whole cube [0, 1]^3, including T + F > 1 (Theorem 8(a)); minimal and unique (8(b)).
    The atom (1,0,1) is a glut: it lies in E_T and in E_F.
    frame = "belnap":  atoms {t, f, b, n} with E_T = {t, b}, E_F = {f, b}, E_I = {n}; faithful exactly on
    {T + I <= 1, F + I <= 1}, which contains part of the paraconsistent zone (Theorem 8(d)).
    frame = "disjoint": atoms {a, iota, not_a}; faithful exactly on T + I + F <= 1 (Theorem 8(d)).
    """

    triple: Triple
    frame: str = "minimal"
    atoms: tuple = field(init=False)
    patterns: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.frame not in _FRAMES:
            raise ValueError(f"frame must be one of {sorted(_FRAMES)}")
        spec = _FRAMES[self.frame]
        self.atoms = tuple(name for name, _ in spec)
        self.patterns = np.array([p for _, p in spec], dtype=float)
        self.triple = tuple(float(v) for v in self.triple)

    # events ------------------------------------------------------------------
    def event(self, component: str) -> np.ndarray:
        """Indicator of E_T, E_I or E_F (``component`` in 'T', 'I', 'F')."""
        return self.patterns[:, "TIF".index(component)]

    @property
    def glut(self) -> np.ndarray:
        """Indicator of the glut atoms E_T intersected with E_F."""
        return self.event("T") * self.event("F")

    def _rows(self):
        return -self.patterns.T, -np.asarray(self.triple)

    def is_empty(self) -> bool:
        A, b = self._rows()
        n = len(self.atoms)
        r = _lp(np.zeros(n), A, b, n)
        return r.status != 0

    def lower(self, event_mask) -> Optional[float]:
        """min P(event) over K; None if K is empty."""
        A, b = self._rows()
        n = len(self.atoms)
        r = _lp(np.asarray(event_mask, float), A, b, n)
        return None if r.status != 0 else float(r.fun)

    def envelope(self) -> Optional[Triple]:
        """(min P(E_T), min P(E_I), min P(E_F)) over K."""
        vals = [self.lower(self.event(c)) for c in "TIF"]
        return None if any(v is None for v in vals) else tuple(vals)

    def represents(self, tol: float = 1e-8) -> bool:
        """Faithful and coherent: K is non-empty and its lower envelope returns the triple exactly."""
        env = self.envelope()
        return env is not None and all(abs(a - b) <= tol for a, b in zip(env, self.triple))

    def glut_lower(self) -> Optional[float]:
        """Least mass K must put on glut atoms. On the minimal frame it is (T + F - 1)^+: the excess
        that the classical frame reads as sure loss (Theorem 3) is carried by the glut."""
        return self.lower(self.glut)

    def extreme_points(self) -> np.ndarray:
        """Minimal frame only: the measures x_k * delta_(1,1,1) + (1 - x_k) * delta_(1 - e_k) of the proof of
        Theorem 8(a); the k-th one attains the lower bound of component k."""
        if self.frame != "minimal":
            raise ValueError("extreme_points is defined for the minimal frame")
        out = np.zeros((3, 4))
        for k, x in enumerate(self.triple):
            out[k, 3] = x
            out[k, k] = 1 - x  # atom k has pattern 1 - e_k
        return out


def to_glut_frame(triple, frame: str = "minimal") -> GlutLifting:
    """Lift a triple (T, I, F) to a credal set on a frame with a glut atom (Theorem 8).

    On the minimal frame every triple of [0, 1]^3 is represented faithfully and coherently, including
    T + F > 1, where the two-atom betting reading incurs sure loss (Theorem 3). Use ``.represents()`` to
    check faithfulness and ``.glut_lower()`` for the mass carried by the glut."""
    t = tuple(float(v) for v in triple)
    if len(t) != 3:
        raise ValueError("triple must have three components")
    if frame == "minimal" and not all(0 <= v <= 1 for v in t):
        raise ValueError("the minimal lifting represents components in [0, 1]")
    return GlutLifting(t, frame)


def glut_conjunction(x, y, dependence: str = "independent", method: str = "closed") -> Triple:
    """Natural extension of the conjunction of two triples on the glut frame (Theorem 9), with
    E_T^{A and B} = E_T^A x E_T^B, E_I^{A and B} = E_I^A u E_I^B, E_F^{A and B} = E_F^A u E_F^B:

    dependence = "independent": (T1 T2, I1 + I2 - I1 I2, F1 + F2 - F1 F2)     (algebraic N-norm, 9(a));
    dependence = "none":        (max(0, T1 + T2 - 1), max(I1, I2), max(F1, F2))  (no assumption, 9(b));
    dependence = "comonotone":  (min(T1, T2), max(I1, I2), max(F1, F2))       (min/max/max N-norm, 9(c)).

    ``method="lp"`` (only for ``"none"``) computes it by linear programming on the 16-atom product frame."""
    (T1, I1, F1), (T2, I2, F2) = (tuple(float(v) for v in t) for t in (x, y))
    if dependence == "independent":
        return T1 * T2, I1 + I2 - I1 * I2, F1 + F2 - F1 * F2
    if dependence == "comonotone":
        return min(T1, T2), max(I1, I2), max(F1, F2)
    if dependence != "none":
        raise ValueError("dependence must be 'independent', 'none' or 'comonotone'")
    if method == "closed":
        return max(0.0, T1 + T2 - 1), max(I1, I2), max(F1, F2)
    pat = np.array([p for _, p in _FRAMES["minimal"]], dtype=float)
    n = 16
    PA = np.repeat(pat, 4, axis=0)   # pattern of the first factor for atom (a, b)
    PB = np.tile(pat, (4, 1))
    A_ub = np.r_[-PA.T, -PB.T]
    b_ub = -np.r_[[T1, I1, F1], [T2, I2, F2]]
    evs = [PA[:, 0] * PB[:, 0],
           np.maximum(PA[:, 1], PB[:, 1]),
           np.maximum(PA[:, 2], PB[:, 2])]
    out = []
    for e in evs:
        r = _lp(e, A_ub, b_ub, n)
        out.append(float(r.fun))
    return tuple(out)


# ============================================================================ retraction to SL (Theorem 5)
def sl_retraction(Ts, Fs) -> Triple:
    """Theorem 5(c): the saturating chart T^s = r / (r + W), F^s = q / (q + W) retracts to the Subjective
    Logic opinion (b, d, u) = (T(1-F), F(1-T), (1-T)(1-F)) / (1 - T F), a bijection onto the non-dogmatic
    opinions of [0, 1)^2. T^s + F^s > 1 holds exactly when r q > W^2; there the betting reading incurs
    sure loss while the evidential reading gives the non-empty IDM interval [b, b + u] (Theorem 5(d))."""
    T, F = float(Ts), float(Fs)
    if not (0 <= T < 1 and 0 <= F < 1):
        raise ValueError("the saturating chart covers [0, 1)^2")
    d = 1 - T * F
    return T * (1 - F) / d, F * (1 - T) / d, (1 - T) * (1 - F) / d


# ============================================================================ one-call diagnosis
def _member_leaf_counts(ens, X) -> np.ndarray:
    """Per-member training class counts in the leaf reached by x, shape (M, n, K)."""
    X = np.asarray(X)
    members = ens.estimator_.estimators_
    feats = getattr(ens.estimator_, "estimators_features_", None)
    out = np.zeros((len(members), X.shape[0], len(ens.classes_)))
    for m, tree in enumerate(members):
        if not hasattr(tree, "tree_"):
            raise ValueError("leaf evidence needs tree members; pass evidence= explicitly")
        Xm = X[:, feats[m]] if feats is not None else X
        leaves = tree.apply(Xm.astype(np.float32))
        v = tree.tree_.value[leaves, 0, :]
        if np.allclose(v.sum(axis=1), 1.0):
            v = v * tree.tree_.weighted_n_node_samples[leaves][:, None]
        out[m][:, ens._member_columns(tree)] += v
    return out


def credal_diagnosis(members=None, *, ensemble=None, X=None, evidence=None, labels=None,
                     events: Optional[Mapping[str, Iterable]] = None, rule: str = "maximality",
                     credal_set: str = "hull", evidence_scale: float = 1.0, W: float = DEFAULT_W):
    """Credal quality checks, natural extension, decision by sets and the RNEL diagnosis in one call.

    Input: ``members`` with shape (M, K) (one instance) or (M, n, K) (member distributions, e.g. the
    probabilities of M classifiers), or a fitted :class:`rnel.credal.CredalEnsemble` with ``X``.

    For each instance it returns a dict with
      * ``triples``: per-class triples (T, I, F) = (min_m p_mk, max - min, 1 - max_m p_mk) of the hull of
        the members (by Theorem 1, I is the credal width);
      * ``status``: 'coherent' / 'incoherent' / 'sure_loss' of the singleton triples (Theorem 2), and
        ``corrected``, their coherent correction (Theorem 1(c));
      * ``events``: the natural extension, as a triple, of each named composite event (Theorem 2(c));
      * ``decision``: the non-dominated classes under ``rule`` ('interval_dominance', 'maximality',
        'e_admissible'), with the credal set taken as the hull of the members (``credal_set="hull"``) or
        as the intervals of the triples (``"intervals"``);
      * ``rnel``: why there is uncertainty, from evidence per source: for each class the state
        (a, b, u, c) of :func:`rnel.credal.evidential_state` (support, rejection, absence of evidence and
        between-source conflict, a + b + u + c = 1) and the multiclass conflict C*.

    Evidence per source: ``evidence`` (V, n, K) if given; for a tree ensemble, the leaf class counts of
    each member divided by M (so the pooled total is ``ensemble.evidence(X, "leaf")``); otherwise each
    member counts as ``evidence_scale`` observations spread by its probabilities.
    The credal part says how wide the set is; the RNEL part says whether the width comes from absence
    of evidence or from sources that disagree."""
    from .credal import c_star_multiclass, evidential_state

    if (members is None) == (ensemble is None):
        raise ValueError("give exactly one of members or ensemble (with X)")
    if ensemble is not None:
        if X is None:
            raise ValueError("X is required with an ensemble")
        P = ensemble.member_proba(X)
        if labels is None:
            labels = list(ensemble.classes_)
        if evidence is None:
            try:
                evidence = _member_leaf_counts(ensemble, X) / P.shape[0]
            except ValueError:
                evidence = None
    else:
        P = np.asarray(members, dtype=float)
    single = P.ndim == 2
    if single:
        P = P[:, None, :]
    if P.ndim != 3:
        raise ValueError("members must have shape (M, K) or (M, n, K)")
    M, n, K = P.shape
    E = evidence_scale * P if evidence is None else np.asarray(evidence, dtype=float)
    if E.ndim == 2:
        E = E[:, None, :]
    lab = labels if labels is not None else list(range(K))
    rules = {"interval_dominance": interval_dominance, "maximality": maximality, "e_admissible": e_admissible}
    if rule not in rules:
        raise ValueError(f"rule must be one of {sorted(rules)}")
    if credal_set not in ("hull", "intervals"):
        raise ValueError("credal_set must be 'hull' or 'intervals'")

    cstar = c_star_multiclass(E, W)
    states = [evidential_state(E, k, W) for k in range(K)]
    reports = []
    for i in range(n):
        Pi = P[:, i, :]
        l, u = Pi.min(axis=0), Pi.max(axis=0)
        trip = np.c_[l, u - l, 1 - u]
        chk = singleton_intervals_check(trip)
        status = "sure_loss" if not chk["avoids_sure_loss"] else ("coherent" if chk["coherent"] else "incoherent")
        corrected = singleton_correction(trip) if status != "sure_loss" else None
        ev = {}
        for name, e in (events or {}).items():
            idx = [lab.index(x) for x in e]
            ev[name] = singleton_natural_extension(trip, idx) if corrected is not None else None
        if credal_set == "hull":
            dec = rules[rule](members=Pi, labels=lab)
        else:
            dec = rules[rule](trip, labels=lab)
        rn = {lab[k]: {key: float(states[k][key][i]) for key in ("a", "b", "u", "c")} for k in range(K)}
        reports.append({
            "triples": {lab[k]: tuple(float(v) for v in trip[k]) for k in range(K)},
            "status": status,
            "corrected": None if corrected is None else {lab[k]: tuple(float(v) for v in corrected[k])
                                                          for k in range(K)},
            "events": ev,
            "rule": rule,
            "decision": dec,
            "rnel": {"per_class": rn, "C_star": float(cstar[i])},
        })
    return reports[0] if single else reports
