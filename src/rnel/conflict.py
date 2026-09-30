"""(Experimental) Between-source conflict from first principles.

Companion code of the dossier "Conflict between sources from first principles: axioms,
identifiability, order invariance and dependent sources". Nothing here changes the published
Definition 8.4 (``rnel.tuple.fused_contradiction``); it adds an alternative that is derived from
axioms on evidence counts.

A *profile* is a finite multiset of sources; source i carries evidence counts (r_i, s_i) for and
against x (``(t, f)`` of a ``Reports``). With margins m_i = r_i - s_i,

    K_w(P) = sum_i min(r_i, s_i)                      (conflict inside sources)
    K_b(P) = min(M+, M-),  M+ = sum_{m_i>0} m_i,  M- = sum_{m_i<0} |m_i|   (between sources)
           = min(R, S) - K_w(P)                         (R, S = pooled counts)
           = sum_i max(r_i, s_i) - max(R, S)            (sup-norm subadditivity gap)

and the normalised component C* = 2 K_b / (R + S + W) (share of the evidence in dispute between
sources, on the scale of Definition 8.1).  The triple (R, S, K_w) is an additive state: fusion of
disjoint sets of sources adds it componentwise, so C* does not depend on the order or bracketing of
fusion.

Dependent sources: reports carry a provenance label; reports with the same label are treated as
copies of, or overlapping draws from, one origin.  ``cautious`` fusion joins them (componentwise
max, idempotent: a copy adds nothing), ``cumulative`` fusion adds them (independence).  Between the
two lies the dependence indeterminacy, returned as an interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Hashable, Iterable, Sequence

from .sl import DEFAULT_W, Opinion, degree_of_conflict
from .tuple import Reports

Counts = tuple[float, float]


def _rs(x) -> Counts:
    if isinstance(x, Reports):
        return float(x.t), float(x.f)
    r, s = x
    return float(r), float(s)


# ----------------------------------------------------------------------------- measures on profiles
def margins(profile: Iterable) -> list[float]:
    return [r - s for r, s in map(_rs, profile)]


def camp_masses(profile: Iterable) -> tuple[float, float]:
    """(M+, M-): total net evidence of the sources leaning towards x and against x."""
    m = margins(profile)
    return sum(v for v in m if v > 0), sum(-v for v in m if v < 0)


def k_within(profile: Iterable) -> float:
    return sum(min(r, s) for r, s in map(_rs, profile))


def k_between(profile: Iterable) -> float:
    """K_b = min(M+, M-) (Theorem 1 of the dossier)."""
    return min(camp_masses(profile))


def credal_gap(profile: Iterable, lower: float, upper: float) -> float:
    """sum_i phi(e_i) - phi(sum_i e_i) with phi the upper expectation over P(x) in [lower, upper].

    Theorem 2 of the dossier: equals (upper - lower) * K_b for every interval."""
    def phi(r, s):
        return max(lower * r + (1 - lower) * s, upper * r + (1 - upper) * s)
    P = [_rs(x) for x in profile]
    R, S = sum(r for r, _ in P), sum(s for _, s in P)
    return sum(phi(r, s) for r, s in P) - phi(R, S)


def sup_gap(vectors: Sequence[Sequence[float]]) -> float:
    """Multinomial between-source conflict: sum_i max_j e_ij - max_j sum_i e_ij (>= 0)."""
    if not vectors:
        return 0.0
    k = len(vectors[0])
    pooled = [sum(v[j] for v in vectors) for j in range(k)]
    return sum(max(v) for v in vectors) - max(pooled)


# ----------------------------------------------------------------------------- additive state
@dataclass(frozen=True)
class ConflictState:
    """Additive sufficient state (R, S, K_w) of a profile. ``a + b`` fuses disjoint profiles."""

    R: float = 0.0
    S: float = 0.0
    Kw: float = 0.0
    n: int = 0  # number of fused sources (bookkeeping only; not needed for C*)

    @classmethod
    def of(cls, source) -> "ConflictState":
        r, s = _rs(source)
        if r < 0 or s < 0:
            raise ValueError("counts must be non-negative (see the dossier for signed counts)")
        return cls(r, s, min(r, s), 1)

    @classmethod
    def of_profile(cls, profile: Iterable) -> "ConflictState":
        st = cls()
        for x in profile:
            st = st + cls.of(x)
        return st

    def __add__(self, other: "ConflictState") -> "ConflictState":
        return ConflictState(self.R + other.R, self.S + other.S, self.Kw + other.Kw, self.n + other.n)

    @property
    def Kb(self) -> float:
        return max(0.0, min(self.R, self.S) - self.Kw)

    @property
    def camps(self) -> tuple[float, float]:
        """(M+, M-) recovered from the state."""
        return self.Kb + max(self.R - self.S, 0.0), self.Kb + max(self.S - self.R, 0.0)

    def c_star(self, W: float = DEFAULT_W) -> float:
        tot = self.R + self.S + W
        return 2 * self.Kb / tot if tot > 0 else 0.0

    def c_within(self, W: float = DEFAULT_W) -> float:
        tot = self.R + self.S + W
        return 2 * self.Kw / tot if tot > 0 else 0.0

    def kappa_between(self) -> float:
        """Scale-free share 2 K_b / (R + S) in [0, 1]."""
        tot = self.R + self.S
        return 2 * self.Kb / tot if tot > 0 else 0.0

    def tuple(self, W: float = DEFAULT_W) -> dict[str, float]:
        """(T, F, G) of the fused counts (Definition 8.1 with c = v = n = 0) and C*."""
        tot = self.R + self.S + W
        return {"T": self.R / tot, "F": self.S / tot, "G": W / tot, "C": self.c_star(W)}

    def opinion(self, W: float = DEFAULT_W, a: float = 0.5) -> Opinion:
        """SL projection: the cumulative-fusion opinion; independent of K_w."""
        return Opinion.from_evidence(self.R, self.S, W, a)


def state_from_tuple(T: float, F: float, G: float, C: float, W: float = DEFAULT_W) -> ConflictState:
    """Identifiability (Proposition 5): (T, F, G, C*) and W determine (R, S, K_b), hence K_w."""
    tot = W / G
    R, S, Kb = T * tot, F * tot, C * tot / 2
    return ConflictState(R, S, min(R, S) - Kb)


def c_star(profile: Iterable, W: float = DEFAULT_W) -> float:
    return ConflictState.of_profile(profile).c_star(W)


# ----------------------------------------------------------------------------- dependent sources
def group_counts(reports: Iterable[tuple[Hashable, object]], mode: str = "cautious") -> list[Counts]:
    """Collapse reports (label, counts) into one source per provenance label.

    mode = "cautious": componentwise max inside a label (idempotent: copies add nothing; the
           smallest count vector compatible with every report of that origin);
    mode = "cumulative": componentwise sum (the reports are treated as independent);
    mode = "average": componentwise mean (SL averaging fusion in evidence form)."""
    groups: dict[Hashable, list[Counts]] = {}
    for g, x in reports:
        groups.setdefault(g, []).append(_rs(x))
    out = []
    for vs in groups.values():
        if mode == "cautious":
            out.append((max(r for r, _ in vs), max(s for _, s in vs)))
        elif mode == "cumulative":
            out.append((sum(r for r, _ in vs), sum(s for _, s in vs)))
        elif mode == "average":
            out.append((sum(r for r, _ in vs) / len(vs), sum(s for _, s in vs) / len(vs)))
        else:
            raise ValueError(mode)
    return out


def dependence_interval(reports: Iterable[tuple[Hashable, object]], W: float = DEFAULT_W) -> dict:
    """Fused tuple under the two extreme dependence assumptions inside each provenance label.

    Returns the cautious and cumulative tuples and I_dep = 1 - Sigma_cautious / Sigma_cumulative,
    the share of the cumulative evidence that may be double-counted."""
    reports = list(reports)
    lo = ConflictState.of_profile(group_counts(reports, "cautious"))
    hi = ConflictState.of_profile(group_counts(reports, "cumulative"))
    s_lo, s_hi = lo.R + lo.S, hi.R + hi.S
    return {"cautious": lo.tuple(W), "cumulative": hi.tuple(W),
            "I_dep": 1 - s_lo / s_hi if s_hi > 0 else 0.0}


def atom_union(reports: Iterable[dict[Hashable, int]]) -> Counts:
    """Exact provenance: each report is a map atom -> +1 (for x) / -1 (against x); fusion is the
    union of atoms (a shared atom counts once)."""
    atoms: dict[Hashable, int] = {}
    for rep in reports:
        for k, v in rep.items():
            if k in atoms and atoms[k] != v:
                raise ValueError(f"atom {k!r} reported with both signs")
            atoms[k] = v
    return float(sum(v > 0 for v in atoms.values())), float(sum(v < 0 for v in atoms.values()))


# ----------------------------------------------------------------------------- Definition 8.4 variants
def def84_sequential(profile: Sequence, W: float = DEFAULT_W, a: float = 0.5) -> float:
    """Binary Definition 8.4 applied left to right: C_{X<>Y} = max(C_X, C_Y, DC(w_X, w_Y)) with C of a
    raw source = 0 (no 'both' reports). Order dependent."""
    P = [_rs(x) for x in profile]
    cur, C = P[0], 0.0
    for x in P[1:]:
        C = max(C, degree_of_conflict(Opinion.from_evidence(*cur, W, a), Opinion.from_evidence(*x, W, a)))
        cur = (cur[0] + x[0], cur[1] + x[1])
    return C


def def84_maxpair(profile: Sequence, W: float = DEFAULT_W, a: float = 0.5) -> float:
    """Multi-source default of Definition 8.4: the largest pairwise DC."""
    ops = [Opinion.from_evidence(*_rs(x), W, a) for x in profile]
    return max((degree_of_conflict(x, y) for x, y in combinations(ops, 2)), default=0.0)
