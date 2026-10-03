"""(Experimental) Neutrosophic statistics from evidence counts: estimates with several labelled I.

No dependencies beyond the standard library.

A refined neutrosophic number is ``N = a + sum_k b_k * I_k`` with every ``I_k`` in [0, 1] and the I_k
mutually independent (the multi-source neutrosophic number, MSNN, of Leyva-Vázquez and Smarandache,
"Neutrosophic interval-indeterminate numbers of the form a + bI", 2026, github.com/mleyvaz/
neutrosophic-affine). Under ``I_k = (1 + e_k) / 2`` it is affine arithmetic (Stolfi and de Figueiredo):
the ranges are the same, shared symbols cancel, and the dependency problem of interval arithmetic
does not arise. What the a + bI form adds is that every symbol is *named*.

Here the names come from the RNEL reports (:class:`rnel.tuple.Reports`): the estimate of the chance of
x is

    p(x) = t / S + (c / S) I_C + (v / S) I_U + (n / S) I_N + (W / S) I_G   (+ extra types I_1, ...)

with S = t + f + c + v + n + sum(extra) + W. I_C, I_U, I_N carry the indeterminacy that comes *from the
data* (contradictory, undetermined and ill-posed reports: observations that exist but cannot be
classified), I_G the indeterminacy that comes from *missing data* (the prior strength W). Each I_k reads
"the share of that mass that goes to x". The estimate of not-x uses the same symbols with 1 - I_k, so

* ``reading="credal"`` (default): p(x) + p(not x) = 1 exactly, and the range of p(x) is the imprecise
  Dirichlet interval with set-valued observations, [t / S, (S - f) / S] (Walley 1996; for c = v = n = 0
  it is ``rnel.neutro_credal.idm_triple``). This reading adds no information to the credal one; it
  adds the decomposition of the width by type and by source.
* ``glut=True``: contradictory reports count for both x and not x (Belnap's "both"), so
  p(x) + p(not x) = 1 + c / S. A single probability cannot represent this.
* ``gap=True``: ill-posed reports count for neither (Belnap's "neither"), p(x) + p(not x) = 1 - n / S.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Hashable, Mapping, Optional, Sequence

from .sl import DEFAULT_W
from .tuple import Reports

__all__ = [
    "NNumber", "NeutroEstimate", "estimate", "estimate_sources", "decompose", "interval_sum",
    "credal_interval", "TYPES",
]

TYPES = ("C", "U", "N")          # data-indeterminacy symbols: contradictory, undetermined, ill-posed
ABSENCE = "G"                     # missing data (prior strength W)
_err_ids = count(1)


def _fresh(prefix: str = "err") -> tuple:
    return (prefix, next(_err_ids))


def _type_of(label) -> Hashable:
    """Type of a symbol: 'C' for 'C' or ('C', source); the label itself otherwise."""
    return label[0] if isinstance(label, tuple) else label


def _source_of(label) -> Optional[Hashable]:
    return label[1] if isinstance(label, tuple) and label[0] != "err" else None


# ============================================================================ the number
class NNumber:
    """Refined neutrosophic number a + sum_k b_k I_k, I_k in [0, 1] independent, k a hashable label."""

    __slots__ = ("a", "terms")

    def __init__(self, a: float, terms: Optional[Mapping[Hashable, float]] = None):
        self.a = float(a)
        self.terms = {k: float(v) for k, v in (terms or {}).items() if v != 0.0}

    # ---- reading
    def range(self) -> tuple[float, float]:
        lo = self.a + sum(min(0.0, v) for v in self.terms.values())
        hi = self.a + sum(max(0.0, v) for v in self.terms.values())
        return lo, hi

    @property
    def lo(self) -> float:
        return self.range()[0]

    @property
    def hi(self) -> float:
        return self.range()[1]

    @property
    def width(self) -> float:
        return sum(abs(v) for v in self.terms.values())

    def at(self, values: Optional[Mapping[Hashable, float]] = None, default: float = 0.5) -> float:
        """Value for given I_k (missing ones take ``default``)."""
        values = values or {}
        return self.a + sum(v * values.get(k, default) for k, v in self.terms.items())

    def __repr__(self) -> str:
        def name(k):
            return f"I_{k[0]}[{k[1]}]" if isinstance(k, tuple) else f"I_{k}"
        return "NNumber(" + " ".join([f"{self.a:.4g}"] + [f"{v:+.4g}*{name(k)}" for k, v in self.terms.items()]) + ")"

    # ---- affine operations (exact, no new symbols)
    @staticmethod
    def _as(x) -> "NNumber":
        return x if isinstance(x, NNumber) else NNumber(float(x))

    def __neg__(self):
        return NNumber(-self.a, {k: -v for k, v in self.terms.items()})

    def __add__(self, other):
        o = self._as(other)
        t = dict(self.terms)
        for k, v in o.terms.items():
            t[k] = t.get(k, 0.0) + v
        return NNumber(self.a + o.a, t)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-self._as(other))

    def __rsub__(self, other):
        return self._as(other) - self

    def __mul__(self, other):
        if not isinstance(other, NNumber):
            c = float(other)
            return NNumber(self.a * c, {k: v * c for k, v in self.terms.items()})
        # Affine product about the centres (I_k = 1/2). With dx = sum b_k (I_k - 1/2), |dx| <= rx =
        # sum |b_k| / 2, the remainder dx * dy lies in [-rx ry, rx ry]: one fresh symbol of
        # coefficient 2 rx ry and a shift of -rx ry (the MSNN of the affine paper used 4 rx ry).
        cx, cy = self.at(), other.at()
        rx, ry = self.width / 2, other.width / 2
        t: dict = {}
        for k, v in self.terms.items():
            t[k] = t.get(k, 0.0) + cy * v
        for k, v in other.terms.items():
            t[k] = t.get(k, 0.0) + cx * v
        a = cx * cy - 0.5 * sum(t.values())
        if rx * ry > 0:
            t[_fresh()] = 2 * rx * ry
            a -= rx * ry
        return NNumber(a, t)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, NNumber):
            raise TypeError("division by an NNumber is not implemented; divide by a scalar")
        return self * (1.0 / float(other))


# ============================================================================ estimates
@dataclass(frozen=True)
class NeutroEstimate:
    """p(x) and p(not x) as refined neutrosophic numbers sharing the same symbols."""

    x: NNumber
    not_x: NNumber
    S: float
    glut: float          # c / S counted twice (glut reading), else 0
    gap: float           # n / S counted for neither (gap reading), else 0

    @property
    def total(self) -> NNumber:
        """p(x) + p(not x): exactly 1 (credal), 1 + glut, or 1 - gap; the symbols cancel."""
        return self.x + self.not_x

    def decomposition(self, by: str = "type") -> dict:
        return decompose(self.x, by=by)


def _split(r: Reports) -> dict:
    d = {"C": r.c, "U": r.v, "N": r.n}
    d.update({f"I{k + 1}": e for k, e in enumerate(r.extra)})
    return d


def _build(t: float, f: float, masses: dict, S: float, glut: bool, gap: bool) -> NeutroEstimate:
    """masses: {label: count} for every symbol, including the absence symbol G."""
    a_x, a_nx = t / S, f / S
    tx, tnx = {}, {}
    g = p = 0.0
    for lab, m in masses.items():
        if m == 0:
            continue
        kind = _type_of(lab)
        if glut and kind == "C":          # supports both: no indeterminacy, counted on both sides
            a_x += m / S
            a_nx += m / S
            g += m / S
        elif gap and kind == "N":         # supports neither
            p += m / S
        else:                             # share I goes to x, 1 - I to not x
            tx[lab] = m / S
            a_nx += m / S
            tnx[lab] = -m / S
    return NeutroEstimate(NNumber(a_x, tx), NNumber(a_nx, tnx), S, g, p)


def estimate(r: Reports, W: float = DEFAULT_W, glut: bool = False, gap: bool = False) -> NeutroEstimate:
    """Neutrosophic estimate of p(x) and p(not x) from one set of reports (one symbol per type)."""
    if W <= 0:
        raise ValueError("W must be positive")
    masses = _split(r)
    masses[ABSENCE] = W
    S = r.t + r.f + sum(masses.values())
    return _build(r.t, r.f, masses, S, glut, gap)


def estimate_sources(sources: Mapping[Hashable, Reports] | Sequence[Reports], W: float = DEFAULT_W,
                     glut: bool = False, gap: bool = False) -> NeutroEstimate:
    """Estimate from several independent sources: counts add (RNEL cumulative fusion) but every data
    symbol stays tied to its source, e.g. ``('C', 'lab A')``; the absence symbol G is shared."""
    items = sources.items() if isinstance(sources, Mapping) else enumerate(sources)
    masses: dict = {}
    t = f = 0.0
    for name, r in items:
        t += r.t
        f += r.f
        for kind, m in _split(r).items():
            masses[(kind, name)] = masses.get((kind, name), 0.0) + m
    masses[ABSENCE] = W
    S = t + f + sum(masses.values())
    return _build(t, f, masses, S, glut, gap)


# ============================================================================ diagnostics
def decompose(x: NNumber, by: str = "type", relative: bool = False) -> dict:
    """Width of x attributed to each symbol, grouped by ``type`` (C, U, N, G, ...), ``source``
    (absence G and linearisation errors are reported under their own keys) or ``symbol``."""
    out: dict = {}
    for k, v in x.terms.items():
        if by == "symbol":
            key = k
        elif by == "type":
            key = _type_of(k)
        elif by == "source":
            src = _source_of(k)
            key = src if src is not None else _type_of(k)
        else:
            raise ValueError("by must be 'type', 'source' or 'symbol'")
        out[key] = out.get(key, 0.0) + abs(v)
    if relative and x.width > 0:
        out = {k: v / x.width for k, v in out.items()}
    return out


def interval_sum(*xs: NNumber) -> tuple[float, float]:
    """What interval arithmetic gives for x_1 + ... + x_m when every operand is replaced by its range
    (the symbols are forgotten). Compare with ``sum(xs).range()``."""
    return sum(x.lo for x in xs), sum(x.hi for x in xs)


def credal_interval(r: Reports, W: float = DEFAULT_W) -> tuple[float, float]:
    """Imprecise Dirichlet interval for x with set-valued (unclassifiable) reports: [t/S, (S - f)/S]."""
    S = r.t + r.f + r.c + r.v + r.n + sum(r.extra) + W
    return r.t / S, (S - r.f) / S
