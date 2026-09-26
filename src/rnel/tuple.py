"""The RNEL evidence tuple (Definition 8.1) and RNEL fusion (Theorem 8.2, Definition 8.4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .sl import DEFAULT_W, Opinion, degree_of_conflict


@dataclass(frozen=True)
class Reports:
    """Report counts about a proposition x.

    t: supporting x; f: against x; c: supporting both (contradictory);
    v: "x or not x, undetermined"; n: "neither" (ill-posed); extra: further indeterminacy types.
    """

    t: float = 0.0
    f: float = 0.0
    c: float = 0.0
    v: float = 0.0
    n: float = 0.0
    extra: tuple[float, ...] = field(default_factory=tuple)

    def __add__(self, other: "Reports") -> "Reports":
        """RNEL cumulative fusion: componentwise addition of counts (independent sources)."""
        k = max(len(self.extra), len(other.extra))
        ex = tuple((self.extra[i] if i < len(self.extra) else 0.0) + (other.extra[i] if i < len(other.extra) else 0.0)
                   for i in range(k))
        return Reports(self.t + other.t, self.f + other.f, self.c + other.c, self.v + other.v, self.n + other.n, ex)


@dataclass(frozen=True)
class RNELTuple:
    """(T, C, U, N, G, I_1, ..., F) of Definition 8.1."""

    T: float
    C: float
    U: float
    N: float
    G: float
    F: float
    I: tuple[float, ...] = field(default_factory=tuple)

    @property
    def total(self) -> float:
        return self.T + self.C + self.U + self.N + self.G + self.F + sum(self.I)

    def as_dict(self) -> dict[str, float]:
        d = {"T": self.T, "C": self.C, "U": self.U, "N": self.N, "G": self.G, "F": self.F}
        d.update({f"I{k + 1}": v for k, v in enumerate(self.I)})
        return d

    def to_sl(self, a: float = 0.5) -> Opinion:
        """Coarsening to SL: T -> b, F -> d, every indeterminacy-type component -> u (retraction if total != 1)."""
        rest = self.total - self.T - self.F
        s = self.T + self.F + rest
        return Opinion(self.T / s, self.F / s, rest / s, a)


def rnel_tuple(r: Reports, W: float = DEFAULT_W) -> RNELTuple:
    """Definition 8.1: every component is its count divided by Sigma = sum of counts + W; G = W / Sigma."""
    S = r.t + r.f + r.c + r.v + r.n + sum(r.extra) + W
    return RNELTuple(r.t / S, r.c / S, r.v / S, r.n / S, W / S, r.f / S, tuple(e / S for e in r.extra))


def sl_opinion(r: Reports, W: float = DEFAULT_W, a: float = 0.5) -> Opinion:
    """The SL opinion from the same reports, keeping only supporting and opposing counts (Theorem 8.2)."""
    return Opinion.from_evidence(r.t, r.f, W, a)


def fused_contradiction(sources: list[Reports], W: float = DEFAULT_W, a: float = 0.5) -> RNELTuple:
    """Definition 8.4 for several sources: T, F, G from the fused counts, C = max(C of each source, largest
    pairwise SL degree of conflict). The total may exceed 1 (paraconsistent)."""
    fused = sources[0]
    for s in sources[1:]:
        fused = fused + s
    base = rnel_tuple(fused, W)
    own = max(rnel_tuple(s, W).C for s in sources)
    ops = [sl_opinion(s, W, a) for s in sources]
    dc = max((degree_of_conflict(x, y) for x, y in combinations(ops, 2)), default=0.0)
    return RNELTuple(base.T, max(own, dc), base.U, base.N, base.G, base.F, base.I)
