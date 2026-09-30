"""Signed evidence and off-opinions (EXPERIMENTAL).

Two layers, deliberately kept apart:

1. EvidenceLedger (sound): reports can be added and later retracted, per source. As long as no source retracts
   more than it reported, the net evidence (r, s) is non-negative and yields an ordinary Subjective Logic
   opinion; removing a source is exactly SL cumulative *unfusion* (Jøsang 2016, ch. 13). A source found to be
   systematically inverted (it reports the opposite of the truth) is handled by `invert`, which moves its
   evidence to the complement: its reports for x become evidence for not-x. This keeps every quantity in [0, 1].

2. OffOpinion (experimental): if more is retracted than was reported (or evidence is scaled beyond a standard),
   the components of the opinion can leave [0, 1]. Such values are returned only on request (allow_off=True),
   always flagged, and are NOT probabilities: when the Beta parameters alpha = r + a W or beta = s + (1 - a) W are
   not positive, there is no Beta distribution behind them. They are useful as diagnostics (over-retraction,
   double counting), not for decisions.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from .sl import DEFAULT_W, Opinion


class OffValueWarning(UserWarning):
    """Raised when a quantity leaves [0, 1] and is returned as an off-value."""


@dataclass(frozen=True)
class OffOpinion:
    """(b, d, u, a) with b + d + u = 1 but components allowed outside [0, 1]."""

    b: float
    d: float
    u: float
    a: float = 0.5
    r: float = 0.0
    s: float = 0.0
    W: float = DEFAULT_W

    @property
    def projected(self) -> float:
        return self.b + self.a * self.u

    @property
    def in_simplex(self) -> bool:
        return min(self.b, self.d, self.u) >= -1e-12

    @property
    def beta_admissible(self) -> bool:
        """True when a Beta(alpha, beta) density exists: alpha = r + aW > 0 and beta = s + (1 - a)W > 0."""
        return self.r + self.a * self.W > 0 and self.s + (1 - self.a) * self.W > 0

    def to_opinion(self) -> Opinion:
        if not self.in_simplex:
            raise ValueError(f"off-opinion {self} is outside the SL simplex; it has no probabilistic reading")
        return Opinion(self.b, self.d, self.u, self.a)


def off_from_evidence(r: float, s: float, W: float = DEFAULT_W, a: float = 0.5) -> OffOpinion:
    """Definition 2.2 of SL applied to signed counts: b = r/S, d = s/S, u = W/S with S = r + s + W (S must be > 0)."""
    S = r + s + W
    if S <= 0:
        raise ValueError("total signed evidence plus W must be positive")
    return OffOpinion(r / S, s / S, W / S, a, r, s, W)


@dataclass
class _Entry:
    reported_for: float = 0.0
    reported_against: float = 0.0
    retracted_for: float = 0.0
    retracted_against: float = 0.0
    inverted: bool = False


@dataclass
class EvidenceLedger:
    """Evidence about one proposition x, kept per source so that it can be retracted or inverted."""

    W: float = DEFAULT_W
    a: float = 0.5
    sources: dict[str, _Entry] = field(default_factory=dict)

    def report(self, source: str, for_x: float = 0.0, against_x: float = 0.0) -> "EvidenceLedger":
        if for_x < 0 or against_x < 0:
            raise ValueError("reports are non-negative counts; use retract() to remove evidence")
        e = self.sources.setdefault(source, _Entry())
        e.reported_for += for_x
        e.reported_against += against_x
        return self

    def retract(self, source: str, for_x: float = 0.0, against_x: float = 0.0) -> "EvidenceLedger":
        """Withdraw reports of a source (e.g. a retracted paper, fake reviews removed)."""
        if for_x < 0 or against_x < 0:
            raise ValueError("retracted amounts are non-negative")
        e = self.sources.setdefault(source, _Entry())
        e.retracted_for += for_x
        e.retracted_against += against_x
        return self

    def retract_source(self, source: str) -> "EvidenceLedger":
        """Remove everything a source reported (SL cumulative unfusion of that source)."""
        e = self.sources[source]
        e.retracted_for, e.retracted_against = e.reported_for, e.reported_against
        return self

    def invert(self, source: str, inverted: bool = True) -> "EvidenceLedger":
        """Mark a source as systematically inverted: its reports for x count as evidence against x and vice
        versa (positive evidence on the complement, instead of negative evidence)."""
        self.sources[source].inverted = inverted
        return self

    def per_source(self) -> dict[str, tuple[float, float]]:
        out = {}
        for name, e in self.sources.items():
            r = e.reported_for - e.retracted_for
            s = e.reported_against - e.retracted_against
            out[name] = (s, r) if e.inverted else (r, s)
        return out

    def over_retracted(self) -> list[str]:
        """Sources that retracted more than they reported (the only way net evidence becomes negative)."""
        return [n for n, (r, s) in self.per_source().items() if r < -1e-12 or s < -1e-12]

    def net(self) -> tuple[float, float]:
        rs = self.per_source().values()
        return sum(r for r, _ in rs), sum(s for _, s in rs)

    def opinion(self, allow_off: bool = False) -> Opinion | OffOpinion:
        r, s = self.net()
        bad = self.over_retracted()
        if not bad:
            return Opinion.from_evidence(r, s, self.W, self.a)
        if not allow_off:
            raise ValueError(f"sources {bad} retracted more than they reported; net evidence would be negative. "
                             "Fix the ledger, invert the source, or call opinion(allow_off=True) for a diagnostic.")
        off = off_from_evidence(r, s, self.W, self.a)
        warnings.warn(f"off-opinion from over-retraction by {bad}: {off}", OffValueWarning, stacklevel=2)
        return off


def cumulative_unfusion(fused: Opinion, removed: Opinion, W: float = DEFAULT_W) -> Opinion:
    """Remove the contribution of `removed` from a cumulative fusion (evidence subtraction). Raises if the
    removed opinion carries more evidence than the fused one on either side."""
    rf, sf = fused.to_evidence(W)
    rr, sr = removed.to_evidence(W)
    r, s = rf - rr, sf - sr
    if r < -1e-9 or s < -1e-9:
        raise ValueError("the removed opinion carries more evidence than the fusion; it was not part of it")
    return Opinion.from_evidence(max(r, 0.0), max(s, 0.0), W, fused.a)


# ================================================================================================
# Section 10 of Smarandache & Leyva-Vázquez (v5), implemented as stated there, for verification.
# ================================================================================================
def offunion(A, B):
    """Definition 10.1: (max T, min I, min F) on [Psi, Omega]."""
    return (max(A[0], B[0]), min(A[1], B[1]), min(A[2], B[2]))


def offintersection(A, B):
    """Definition 10.1: (min T, max I, max F)."""
    return (min(A[0], B[0]), max(A[1], B[1]), max(A[2], B[2]))


def offcomplement(A, psi: float = 0.0, omega: float = 1.0):
    """Definition 10.1: C(A) = (F, Psi + Omega - I, T)."""
    return (A[2], psi + omega - A[1], A[0])


def scaled_and(x, y, sigma: float):
    """Theorem 10.4: x AND_sigma y = sigma^-1 (x AND y) on the over-plane T + I + N + F = sigma."""
    from .operators import tinf_and
    return {k: v / sigma for k, v in tinf_and(x, y).items()}


def scaled_or(x, y, sigma: float):
    from .operators import tinf_or
    return {k: v / sigma for k, v in tinf_or(x, y).items()}


def typed_discount(p: float, x: dict, link: str) -> dict:
    """Definition 6.3 for any real trust p (Theorem 10.5): every component X -> pX, then U_link += 1 - p."""
    out = {k: p * v for k, v in x.items()}
    out[link] = out.get(link, 0.0) + 1 - p
    return out


def off_rnel_tuple(t=0.0, f=0.0, c=0.0, v=0.0, n=0.0, W: float = DEFAULT_W, **ik) -> dict:
    """Definition 10.7: Definition 8.1 with signed counts (reports minus retracted reports)."""
    S = t + f + c + v + n + sum(ik.values()) + W
    if S <= 0:
        raise ValueError("Sigma must be positive")
    out = {"T": t / S, "C": c / S, "U": v / S, "N": n / S, "G": W / S, "F": f / S}
    out.update({k: val / S for k, val in ik.items()})
    return out


def retraction_nu(T: float, I: float, F: float, a: float = 0.5) -> tuple[float, float, float, float]:
    """Definition 4.2: (T, F, I + 1 - S) if S <= 1, (T/S, F/S, I/S) if S > 1; returns (b, d, u, a)."""
    S = T + I + F
    if S <= 1:
        return T, F, I + 1 - S, a
    return T / S, F / S, I / S, a


def projected_via_nu(tuple_: dict, a: float = 0.5) -> float:
    """Definition 4.4: P(x) = b_nu + a u_nu after coarsening (all indeterminacy-type components into I)."""
    T, F = tuple_["T"], tuple_["F"]
    I = sum(v for k, v in tuple_.items() if k not in ("T", "F"))
    b, d, u, _ = retraction_nu(T, I, F, a)
    return b + a * u


def signed_multiply(x, y, a: float = 0.0):
    """SL multiplication (Jøsang) on (b, d, u) triples without range checks, same base rate a for both."""
    bx, dx, ux = x
    by, dy, uy = y
    den = 1 - a * a
    b = bx * by + ((1 - a) * a * bx * uy + a * (1 - a) * ux * by) / den
    d = dx + dy - dx * dy
    return b, d, 1 - b - d


def signed_comultiply(x, y, a: float = 1.0):
    bx, dx, ux = x
    by, dy, uy = y
    den = a + a - a * a
    b = bx + by - bx * by
    d = dx * dy + (a * (1 - a) * dx * uy + (1 - a) * a * ux * dy) / den
    return b, d, 1 - b - d


def complement_discount(p: float, b: float, d: float, u: float, a: float = 0.5):
    """Alternative to discounting with p < 0 (not in the paper): a source trusted at level |p| whose reports are
    inverted contributes its opinion about the complement, (d, b, u), discounted by |p|. Stays in the simplex."""
    q = abs(p)
    x = (d, b, u) if p < 0 else (b, d, u)
    bb, dd = q * x[0], q * x[1]
    return bb, dd, 1 - bb - dd
