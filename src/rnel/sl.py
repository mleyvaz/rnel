"""Subjective Logic (Jøsang, 2016): binomial opinions and the operators used by RNEL."""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_W = 2.0


@dataclass(frozen=True)
class Opinion:
    """Binomial opinion (b, d, u, a) with b + d + u = 1."""

    b: float
    d: float
    u: float
    a: float = 0.5

    def __post_init__(self):
        if min(self.b, self.d, self.u) < -1e-12 or abs(self.b + self.d + self.u - 1) > 1e-9:
            raise ValueError(f"invalid opinion {self}")

    @property
    def projected(self) -> float:
        """Projected probability P = b + a u."""
        return self.b + self.a * self.u

    @classmethod
    def from_evidence(cls, r: float, s: float, W: float = DEFAULT_W, a: float = 0.5) -> "Opinion":
        """Definition 2.2: b = r/(r+s+W), d = s/(r+s+W), u = W/(r+s+W)."""
        S = r + s + W
        return cls(r / S, s / S, W / S, a)

    def to_evidence(self, W: float = DEFAULT_W) -> tuple[float, float]:
        if self.u <= 0:
            raise ValueError("dogmatic opinion has infinite evidence")
        return W * self.b / self.u, W * self.d / self.u

    def complement(self) -> "Opinion":
        return Opinion(self.d, self.b, self.u, 1 - self.a)


def multiply(x: Opinion, y: Opinion) -> Opinion:
    """SL multiplication (AND of independent propositions)."""
    den = 1 - x.a * y.a
    b = x.b * y.b + ((1 - x.a) * y.a * x.b * y.u + x.a * (1 - y.a) * x.u * y.b) / den if den > 0 else x.b * y.b
    d = x.d + y.d - x.d * y.d
    u = 1 - b - d
    return Opinion(b, d, u, x.a * y.a)


def comultiply(x: Opinion, y: Opinion) -> Opinion:
    """SL comultiplication (OR of independent propositions)."""
    den = x.a + y.a - x.a * y.a
    b = x.b + y.b - x.b * y.b
    d = x.d * y.d + (x.a * (1 - y.a) * x.d * y.u + (1 - x.a) * y.a * x.u * y.d) / den if den > 0 else x.d * y.d
    u = 1 - b - d
    return Opinion(b, d, u, den)


def cumulative_fusion(A: Opinion, B: Opinion) -> Opinion:
    """Cumulative fusion of independent sources (adds their evidence)."""
    k = A.u + B.u - A.u * B.u
    if k == 0:  # two dogmatic opinions: average
        return Opinion((A.b + B.b) / 2, (A.d + B.d) / 2, 0.0, (A.a + B.a) / 2)
    b = (A.b * B.u + B.b * A.u) / k
    d = (A.d * B.u + B.d * A.u) / k
    # Fused base rate (Josang 2016, ch. 12): weights each source's base rate by the other's
    # uncertainty; equals the mean when u_A = u_B, and returns B's base rate when A is vacuous.
    den_a = A.u + B.u - 2 * A.u * B.u
    a = ((A.a * B.u + B.a * A.u - (A.a + B.a) * A.u * B.u) / den_a) if den_a > 1e-15 else (A.a + B.a) / 2
    return Opinion(b, d, A.u * B.u / k, a)


def averaging_fusion(A: Opinion, B: Opinion) -> Opinion:
    """Averaging fusion of dependent sources."""
    k = A.u + B.u
    if k == 0:
        return Opinion((A.b + B.b) / 2, (A.d + B.d) / 2, 0.0, (A.a + B.a) / 2)
    b = (A.b * B.u + B.b * A.u) / k
    d = (A.d * B.u + B.d * A.u) / k
    return Opinion(b, d, 2 * A.u * B.u / k, (A.a + B.a) / 2)


def discount(trust: float, x: Opinion) -> Opinion:
    """Trust discounting with projected trust p: (p b, p d, p u + 1 - p)."""
    return Opinion(trust * x.b, trust * x.d, trust * x.u + 1 - trust, x.a)


def age(x: Opinion, lam: float, W: float = DEFAULT_W) -> Opinion:
    """Evidence ageing: (r, s) -> (lam r, lam s)."""
    r, s = x.to_evidence(W)
    return Opinion.from_evidence(lam * r, lam * s, W, x.a)


def degree_of_conflict(A: Opinion, B: Opinion) -> float:
    """DC = |P_A - P_B| (1 - u_A)(1 - u_B)."""
    return abs(A.projected - B.projected) * (1 - A.u) * (1 - B.u)
