"""rnel.scores (experimental): score functions of neutrosophic triples as decisions under an explicit base rate.

Book reference: section "Base rates and score functions" (after Theorem 5). The algebraic results are formally
verified in Lean 4 + Mathlib (``formal/lean_scores/LeanScores/Basic.lean`` and
``formal/NeutroEvidence/NeutroEvidence/BaseRate.lean``) and checked with Z3 (``formal/z3/verify_scores_z3.py``);
``tests/test_scores.py`` mirrors each Lean theorem numerically.

For a triple (T, I, F) and a base rate a in [0, 1]

    S_a(T, F)            = (1 - a) T + a (1 - F)            Hurwicz criterion on the interval [T, 1 - F]
    S_{a,lam}(T, I, F)   = S_a(T, F) - lam I                with an explicit aversion lam to indeterminacy
    classic(T, I, F)     = (2 + T - I - F) / 3              = (2/3) S_{1/2} + (1 - I)/3  (Lean: classic_decomp)

In the normalised case T + I + F = 1, S_a = T + a I, the projected probability b + a u of a Subjective Logic
opinion (b, d, u) = (T, F, I) with base rate a (Theorem 5); the classic score then ranks exactly as T.
The single formula S_a = T - a (T + F - 1) covers the gap case (T + F <= 1: T <= S_a <= 1 - F) and the glut case
(T + F >= 1: 1 - F <= S_a <= T). With an imprecise base rate a in [aL, aU] the induced order is partial:
x dominates y for every a in the interval iff it does at both endpoints (Lean: dominance_interval); when no
alternative dominates all others, the honest output is an abstention.
"""
from __future__ import annotations

from typing import Hashable, List, Mapping, Optional, Sequence, Tuple, Union

Triple = Tuple[float, float, float]
Interval = Tuple[float, float]

__all__ = [
    "S", "S_lambda", "classic_score", "accuracy", "hurwicz", "projected_probability", "reading",
    "S_bounds", "score_interval", "dominates", "dominance_pairs", "undominated", "decide",
]


def _check_a(a: float) -> None:
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"base rate must lie in [0, 1], got {a}")


def S(a: float, T: float, F: float) -> float:
    """S_a = (1 - a) T + a (1 - F): the score with base rate a (Lean ``S``)."""
    return (1 - a) * T + a * (1 - F)


def S_lambda(a: float, lam: float, T: float, I: float, F: float) -> float:
    """S_{a,lam} = S_a - lam I (Lean ``Sl``). In the normalised case it equals T + (a - lam) I."""
    return S(a, T, F) - lam * I


def classic_score(T: float, I: float, F: float) -> float:
    """The classic score (2 + T - I - F)/3 (Lean ``classic``); equals (2/3) S_{1/2,1/2} + 1/3."""
    return (2 + T - I - F) / 3


def accuracy(T: float, F: float) -> float:
    """Accuracy function T - F; ranks exactly as S_{1/2} (Lean ``accuracy_ranks_as_S_half``)."""
    return T - F


def hurwicz(lower: float, upper: float, alpha: float) -> float:
    """Hurwicz criterion (1 - alpha) * lower + alpha * upper with optimism alpha. S_a = hurwicz(T, 1 - F, a)."""
    return (1 - alpha) * lower + alpha * upper


def projected_probability(T: float, I: float, a: float) -> float:
    """T + a I: the Subjective Logic projected probability b + a u of the opinion (b, d, u) = (T, F, I)."""
    return T + a * I


def reading(T: float, F: float, tol: float = 1e-12) -> str:
    """'gap' if T + F < 1 (interval [T, 1-F]), 'glut' if T + F > 1 (reversed interval), 'exact' if T + F = 1."""
    s = T + F - 1
    if s < -tol:
        return "gap"
    if s > tol:
        return "glut"
    return "exact"


def S_bounds(T: float, F: float) -> Interval:
    """Range of S_a over a in [0, 1]: [T, 1-F] in the gap case and [1-F, T] in the glut case
    (Lean ``S_gap_bounds``, ``S_glut_bounds``)."""
    return (min(T, 1 - F), max(T, 1 - F))


def score_interval(t: Sequence[float], a_interval: Interval, lam: float = 0.0) -> Interval:
    """Range of S_{a,lam}(t) for a in [aL, aU] (S is affine in a, so the range is attained at the endpoints)."""
    aL, aU = a_interval
    _check_a(aL); _check_a(aU)
    T, I, F = t
    v = (S_lambda(aL, lam, T, I, F), S_lambda(aU, lam, T, I, F))
    return (min(v), max(v))


def dominates(x: Sequence[float], y: Sequence[float], a_interval: Union[float, Interval],
              lam: float = 0.0, strict: bool = False, tol: float = 1e-12) -> bool:
    """True when S_{a,lam}(x) >= S_{a,lam}(y) for every a in [aL, aU] (strict: > at some a as well).

    It suffices to check the two endpoints (Lean ``dominance_interval``). A single number is a precise base rate."""
    aL, aU = (a_interval, a_interval) if isinstance(a_interval, (int, float)) else a_interval
    _check_a(aL); _check_a(aU)
    if aL > aU:
        raise ValueError("a_interval must satisfy aL <= aU")
    d = [S_lambda(a, lam, *x) - S_lambda(a, lam, *y) for a in (aL, aU)]
    if min(d) < -tol:
        return False
    return max(d) > tol if strict else True


def _items(alternatives) -> List[Tuple[Hashable, Triple]]:
    if isinstance(alternatives, Mapping):
        return [(k, tuple(v)) for k, v in alternatives.items()]
    return [(i, tuple(v)) for i, v in enumerate(alternatives)]


def dominance_pairs(alternatives, a_interval: Union[float, Interval], lam: float = 0.0) -> List[Tuple[Hashable, Hashable]]:
    """All pairs (i, j) such that alternative i strictly dominates j over the base-rate interval (a partial order)."""
    it = _items(alternatives)
    return [(i, j) for i, x in it for j, y in it if i != j and dominates(x, y, a_interval, lam, strict=True)]


def undominated(alternatives, a_interval: Union[float, Interval], lam: float = 0.0) -> List[Hashable]:
    """Alternatives not strictly dominated by any other over the base-rate interval."""
    beaten = {j for _, j in dominance_pairs(alternatives, a_interval, lam)}
    return [i for i, _ in _items(alternatives) if i not in beaten]


def decide(alternatives, a_interval: Union[float, Interval], lam: float = 0.0) -> Tuple[Optional[Hashable], List[Hashable]]:
    """(choice, undominated set): the choice is the unique alternative that dominates every other one over the
    interval, or None (abstain) when the base-rate imprecision leaves the decision open."""
    it = _items(alternatives)
    und = undominated(alternatives, a_interval, lam)
    for i, x in it:
        if all(dominates(x, y, a_interval, lam) for j, y in it if j != i):
            if len(und) == 1 or all(dominates(x, y, a_interval, lam, strict=True) for j, y in it if j != i):
                return i, und
    return None, und
