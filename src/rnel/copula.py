"""rnel.copula (experimental): copulas, copula N-norms and the choice of an N-norm as a dependence model.

Book reference: Section 8.3 ("Choosing an N-norm in practice: dependence, copulas and robustness") and the
copula part of Theorem 10 (Chapter 10). Pure Python (``math`` only).

A copula ``C(u, v)`` on [0, 1]^2 is a dependence model between two events with probabilities u and v. The
N-norm induced by C on the glut frame is

    N_C(x1, x2) = (C(T1, T2), I1 + I2 - C(I1, I2), F1 + F2 - C(F1, F2)).

Contents
  pi_copula, upper_frechet (M), lower_frechet (W)     the three basic copulas
  frank(theta), clayton(theta), gumbel(theta), fgm(theta), survival(C), shuffle_S()
  copula(name, theta=None)                              look-up by name ("Pi", "M", "W", "Frank", ...)
  is_radially_symmetric(C)                              numerical check of C == survival(C) on a grid
  nnorm(x, y, C), nnorm_n(xs, Cs)                       copula N-norm, n-fold composition
  frechet_bracket(xs)                                   (N_W, N_M) of Proposition 8.3.1
  surplus(C, x, y)                                      sigma_C of Section 8.3.3 (classical-frame surplus)
  surplus_W_closed_form(x, y)                           closed form used in Proposition 8.3.4
  classic_score(t)                                      s = (2 + T - I - F)/3 (monotone score of Section 8.3.4)
  tail_dependence(C, which)                             numerical lambda_L / lambda_U
  kendall_tau(family, theta), frank_theta_from_tau(tau)

Results covered by tests (``tests/test_copula.py``): Propositions 8.3.1-8.3.4 and Table 8.3.3.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, Iterable, Sequence, Tuple

Copula = Callable[[float, float], float]
Triple = Tuple[float, float, float]

__all__ = [
    "pi_copula", "upper_frechet", "lower_frechet", "frank", "clayton", "gumbel", "fgm", "survival", "shuffle_S",
    "copula", "is_radially_symmetric", "RADIALLY_SYMMETRIC", "nnorm", "nnorm_n", "frechet_bracket", "surplus",
    "surplus_W_closed_form", "classic_score", "tail_dependence", "kendall_tau", "frank_theta_from_tau",
]


# ----------------------------------------------------------------------------- basic copulas
def pi_copula(u: float, v: float) -> float:
    """Product copula Pi(u, v) = uv (independence)."""
    return u * v


def upper_frechet(u: float, v: float) -> float:
    """Upper Fréchet copula M(u, v) = min(u, v) (comonotone dependence)."""
    return min(u, v)


def lower_frechet(u: float, v: float) -> float:
    """Lower Fréchet (Łukasiewicz) copula W(u, v) = max(0, u + v - 1) (countermonotone dependence)."""
    return max(0.0, u + v - 1.0)


def frank(theta: float) -> Copula:
    """Frank copula with parameter theta (theta = 0 is Pi; -inf and +inf are the limits W and M).

    Uses the numerically stable form of the book's script ``nnorm_choice.py`` and the reflection
    C_{-t}(u, v) = u - C_t(u, 1 - v) for negative parameters.
    """
    if theta == 0:
        return pi_copula

    def cpos(u, v, t):
        if u <= 0 or v <= 0:
            return 0.0
        m, mx = min(u, v), max(u, v)
        arg = (1 + math.exp(-t * (mx - m)) - math.exp(-t * mx) - math.exp(-t * (1 - m))) / (-math.expm1(-t))
        return min(max(m - math.log(arg) / t, max(0.0, u + v - 1)), m)

    def C(u, v):
        if theta > 0:
            return cpos(u, v, theta)
        return min(max(u - cpos(u, 1 - v, -theta), max(0.0, u + v - 1)), min(u, v))

    C.__name__ = f"frank({theta:g})"
    return C


def clayton(theta: float) -> Copula:
    """Clayton copula (theta > 0). Not radially symmetric."""
    if theta <= 0:
        raise ValueError("clayton: theta must be positive")

    def C(u, v):
        if u <= 0 or v <= 0:
            return 0.0
        return max(u ** -theta + v ** -theta - 1.0, 0.0) ** (-1.0 / theta)

    C.__name__ = f"clayton({theta:g})"
    return C


def gumbel(theta: float) -> Copula:
    """Gumbel copula (theta >= 1). Not radially symmetric for theta > 1."""
    if theta < 1:
        raise ValueError("gumbel: theta must be >= 1")

    def C(u, v):
        if u <= 0 or v <= 0:
            return 0.0
        return math.exp(-((-math.log(u)) ** theta + (-math.log(v)) ** theta) ** (1.0 / theta))

    C.__name__ = f"gumbel({theta:g})"
    return C


def fgm(theta: float) -> Copula:
    """Farlie-Gumbel-Morgenstern copula, theta in [-1, 1]. Radially symmetric."""
    if not -1 <= theta <= 1:
        raise ValueError("fgm: theta must lie in [-1, 1]")

    def C(u, v):
        return u * v * (1 + theta * (1 - u) * (1 - v))

    C.__name__ = f"fgm({theta:g})"
    return C


def survival(C: Copula) -> Copula:
    """Survival copula Ĉ(u, v) = u + v - 1 + C(1 - u, 1 - v)."""
    def Ch(u, v):
        return u + v - 1.0 + C(1.0 - u, 1.0 - v)

    Ch.__name__ = f"survival({getattr(C, '__name__', 'C')})"
    return Ch


def shuffle_S() -> Copula:
    """The straight shuffle of M with four strips and permutation (0, 2, 1, 3) of Proposition 8.3.3.

    Accepts floats or ``fractions.Fraction`` (exact arithmetic with Fraction inputs).
    """
    segs = [(Fraction(0), Fraction(0)), (Fraction(1, 4), Fraction(1, 4)),
            (Fraction(1, 2), Fraction(-1, 4)), (Fraction(3, 4), Fraction(0))]

    def S(u, v):
        exact = isinstance(u, Fraction) and isinstance(v, Fraction)
        u_, v_ = (u, v) if exact else (Fraction(u).limit_denominator(10 ** 12), Fraction(v).limit_denominator(10 ** 12))
        tot = Fraction(0)
        for a, s in segs:
            hi = min(a + Fraction(1, 4), u_, v_ - s)
            if hi > a:
                tot += hi - a
        return tot if exact else float(tot)

    S.__name__ = "shuffle_S"
    return S


_FAMILIES = {"frank": frank, "clayton": clayton, "gumbel": gumbel, "fgm": fgm}
_BASIC = {"pi": pi_copula, "product": pi_copula, "independence": pi_copula,
          "m": upper_frechet, "min": upper_frechet, "comonotone": upper_frechet,
          "w": lower_frechet, "lukasiewicz": lower_frechet, "countermonotone": lower_frechet}

#: copulas known (from the literature, Nelsen 2006) to be radially symmetric
RADIALLY_SYMMETRIC = ("pi", "m", "w", "frank", "fgm", "shuffle_S")


def copula(name: str, theta: float | None = None) -> Copula:
    """Return a copula by name: "Pi", "M", "W" (and aliases), or a family "Frank", "Clayton", "Gumbel", "FGM"
    with parameter ``theta``. A survival copula is obtained with the prefix "survival:" (e.g. "survival:Clayton")."""
    key = name.strip().lower()
    if key.startswith("survival:"):
        return survival(copula(name.split(":", 1)[1], theta))
    if key in _BASIC:
        return _BASIC[key]
    if key in _FAMILIES:
        if theta is None:
            raise ValueError(f"copula {name!r} needs a parameter theta")
        return _FAMILIES[key](theta)
    raise ValueError(f"unknown copula {name!r}")


def is_radially_symmetric(C: Copula, n: int = 40, tol: float = 1e-9) -> bool:
    """Numerical check of C == Ĉ on the grid of step 1/n (a necessary condition only)."""
    Ch = survival(C)
    g = [i / n for i in range(n + 1)]
    return max(abs(C(u, v) - Ch(u, v)) for u in g for v in g) <= tol


# ----------------------------------------------------------------------------- N-norms
def nnorm(x: Sequence[float], y: Sequence[float], C: Copula = pi_copula) -> Triple:
    """Copula N-norm N_C(x, y) = (C(T1,T2), I1+I2-C(I1,I2), F1+F2-C(F1,F2)) (Section 8.3.1)."""
    (T1, I1, F1), (T2, I2, F2) = x, y
    return (C(T1, T2), I1 + I2 - C(I1, I2), F1 + F2 - C(F1, F2))


def nnorm_n(xs: Sequence[Sequence[float]], Cs: Copula | Sequence[Copula] = pi_copula) -> Triple:
    """n-fold composition N_{C_{n-1}}(... N_{C_1}(x1, x2) ..., xn); ``Cs`` is one copula or n-1 copulas."""
    xs = [tuple(map(float, x)) for x in xs]
    if len(xs) < 2:
        raise ValueError("nnorm_n needs at least two triples")
    if callable(Cs):
        Cs = [Cs] * (len(xs) - 1)
    if len(Cs) != len(xs) - 1:
        raise ValueError("nnorm_n: need len(xs) - 1 copulas")
    out = xs[0]
    for C, x in zip(Cs, xs[1:]):
        out = nnorm(out, x, C)
    return out


def frechet_bracket(xs: Sequence[Sequence[float]]) -> Tuple[Triple, Triple]:
    """(N_W, N_M) for the n-fold composition: every copula N-norm lies componentwise between them,
    T_W <= T_N <= T_M, I_M <= I_N <= I_W, F_M <= F_N <= F_W (Proposition 8.3.1)."""
    return nnorm_n(xs, lower_frechet), nnorm_n(xs, upper_frechet)


def surplus(C: Copula, x: Sequence[float], y: Sequence[float]) -> float:
    """sigma_C(x, y) = I1 + I2 - C(I1, I2) - C(1-F1, 1-F2) + C(T1, T2): the indeterminacy of N_C minus the width
    of the credal conjunction under the same copula on the classical frame (normalised inputs, Section 8.3.3).
    C is *conservative* when sigma_C >= 0 for all pairs of normalised triples."""
    (T1, I1, F1), (T2, I2, F2) = x, y
    return I1 + I2 - C(I1, I2) - C(1 - F1, 1 - F2) + C(T1, T2)


def surplus_W_closed_form(x: Sequence[float], y: Sequence[float]) -> float:
    """sigma_W = a - (a-1)^+ - (t+a)^+ + t^+ with a = I1 + I2 and t = T1 + T2 - 1 (proof of Proposition 8.3.4)."""
    a = x[1] + y[1]
    t = x[0] + y[0] - 1
    pos = lambda z: max(z, 0.0)
    return a - pos(a - 1) - pos(t + a) + pos(t)


def classic_score(t: Sequence[float]) -> float:
    """s(T, I, F) = (2 + T - I - F)/3, the monotone score of Section 8.3.4 (see also ``rnel.scores``)."""
    T, I, F = t
    return (2 + T - I - F) / 3


def tail_dependence(C: Copula, which: str = "lower", t: float = 1e-6) -> float:
    """Numerical tail-dependence coefficient: lower C(t,t)/t, upper (1 - 2(1-t) + C(1-t,1-t))/t."""
    if which == "lower":
        return C(t, t) / t
    if which == "upper":
        return (1 - 2 * (1 - t) + C(1 - t, 1 - t)) / t
    raise ValueError("which must be 'lower' or 'upper'")


def _debye1(x: float) -> float:
    if x == 0:
        return 1.0
    ax = abs(x)
    n = 2000  # Simpson on [0, ax] of t/(e^t - 1)
    h = ax / n
    f = lambda s: 1.0 if s == 0 else s / math.expm1(s)
    tot = f(0) + f(ax) + sum((4 if k % 2 else 2) * f(k * h) for k in range(1, n))
    val = tot * h / 3 / ax
    return val if x > 0 else val + ax / 2


def kendall_tau(family: str, theta: float) -> float:
    """Kendall's tau of a one-parameter family: Clayton theta/(theta+2), Gumbel 1 - 1/theta,
    Frank 1 - 4/theta (1 - D1(theta)), FGM 2 theta / 9."""
    f = family.lower()
    if f == "clayton":
        return theta / (theta + 2)
    if f == "gumbel":
        return 1 - 1 / theta
    if f == "fgm":
        return 2 * theta / 9
    if f == "frank":
        return 0.0 if theta == 0 else 1 - 4 / theta * (1 - _debye1(theta))
    raise ValueError(f"unknown family {family!r}")


def frank_theta_from_tau(tau: float, lo: float = -200.0, hi: float = 200.0) -> float:
    """Invert Kendall's tau for the Frank family by bisection (|tau| < 0.97)."""
    if abs(tau) < 1e-12:
        return 0.0
    if abs(tau) >= 0.97:
        return float("nan")
    g = lambda th: kendall_tau("frank", th) - tau
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if m == 0:
            m = 1e-9
        if g(a) * g(m) <= 0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)
