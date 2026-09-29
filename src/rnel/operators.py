"""Priority-product operators (Sections 5.2, 6.2 and 8.3 of Smarandache and Leyva-Vázquez).

A tuple is a dict {component: value}. Conjunction multiplies every pair of components and assigns each
product to the component of higher priority; components in the same tier share a product equally.
Disjunction uses the reverse order and equals the De Morgan dual of conjunction.
"""
from __future__ import annotations

from typing import Mapping

Tuple = dict[str, float]


def priority_product(x: Mapping[str, float], y: Mapping[str, float], tiers: list[list[str]]) -> Tuple:
    """Conjunction-type product with tiers ordered from lowest to highest priority."""
    rank = {c: i for i, tier in enumerate(tiers) for c in tier}
    out = {c: 0.0 for c in rank}
    for a, xa in x.items():
        for b, yb in y.items():
            p = xa * yb
            if rank[a] != rank[b]:
                out[a if rank[a] > rank[b] else b] += p
            elif a == b:
                out[a] += p
            else:
                out[a] += p / 2
                out[b] += p / 2
    return out


def mass(x: Mapping[str, float]) -> float:
    return sum(x.values())


# ----------------------------------------------------------------- (T, I, N, F), Section 5.2
TINF_AND = [["T"], ["N"], ["I"], ["F"]]
TINF_OR = [["F"], ["I"], ["N"], ["T"]]


def tinf_not(x: Mapping[str, float]) -> Tuple:
    """Negation kept from (T, I, N, F): (F, N, I, T)."""
    return {"T": x["F"], "I": x["N"], "N": x["I"], "F": x["T"]}


def tinf_and(x, y) -> Tuple:
    return priority_product(x, y, TINF_AND)


def tinf_or(x, y) -> Tuple:
    return priority_product(x, y, TINF_OR)


# ----------------------------------------------------------------- (T, I, N, U1..Un, F), Section 6.2
def mu_tiers(n: int, disjunction: bool = False) -> list[list[str]]:
    t = [["T"], ["N"], [f"U{k}" for k in range(1, n + 1)], ["I"], ["F"]]
    return t[::-1] if disjunction else t


def mu_not(x: Mapping[str, float]) -> Tuple:
    """(T, I, N, U1..Un, F) -> (F, N, I, U1..Un, T)."""
    out = dict(x)
    out["T"], out["F"], out["I"], out["N"] = x["F"], x["T"], x["N"], x["I"]
    return out


def mu_and(x, y) -> Tuple:
    n = sum(1 for k in x if k.startswith("U"))
    return priority_product(x, y, mu_tiers(n))


def mu_or(x, y) -> Tuple:
    n = sum(1 for k in x if k.startswith("U"))
    return priority_product(x, y, mu_tiers(n, disjunction=True))


# ----------------------------------------------------------------- RNEL, Section 8.3
def rnel_tiers(components: list[str], disjunction: bool = False) -> list[list[str]]:
    middle = [c for c in components if c not in ("T", "F")]
    t = [["T"], middle, ["F"]]
    return t[::-1] if disjunction else t


def rnel_not(x: Mapping[str, float]) -> Tuple:
    """Only T and F swap: contradiction, undetermined, neither and ignorance are fixed points (Belnap)."""
    out = dict(x)
    out["T"], out["F"] = x["F"], x["T"]
    return out


def rnel_and(x, y) -> Tuple:
    return priority_product(x, y, rnel_tiers(list(x)))


def rnel_or(x, y) -> Tuple:
    return priority_product(x, y, rnel_tiers(list(x), disjunction=True))


def implies(x, y, neg, disj) -> Tuple:
    """x -> y := not x or y."""
    return disj(neg(x), y)


def coarsen(x: Mapping[str, float]) -> tuple[float, float, float]:
    """pi: (T, everything else, F) -> SL (b, u, d) coordinates, unnormalised."""
    rest = mass(x) - x["T"] - x["F"]
    return x["T"], rest, x["F"]


# ----------------------------------------------------------------- base-rate-calibrated versions, Section 5.2
def _indeterminacy(x: Mapping[str, float]) -> float:
    return mass(x) - x["T"] - x["F"]


def _transfer(z: Tuple, k: float, to: str) -> Tuple:
    """Move mass k from the indeterminacy components of z (all but T and F) to component `to`,
    taking from each in proportion to its size."""
    out = dict(z)
    rest = [c for c in z if c not in ("T", "F")]
    tot = sum(z[c] for c in rest)
    if k == 0 or tot == 0:
        return out
    for c in rest:
        out[c] -= k * z[c] / tot
    out[to] += k
    return out


def calibrated_and(x, y, ax: float, ay: float, conj=tinf_and) -> Tuple:
    """Base-rate-calibrated conjunction: the priority-product conjunction `conj` (tinf_and, mu_and or rnel_and)
    plus the transfer K = [(1-ax) ay T_x I_y + ax (1-ay) I_x T_y] / (1 - ax ay) from indeterminacy to T, where
    I is the total indeterminacy mass. On normalised inputs its coarsening is SL multiplication with base
    rates ax, ay. The base rate of the result is ax * ay."""
    z = conj(x, y)
    den = 1 - ax * ay
    if den <= 0:
        return z
    k = ((1 - ax) * ay * x["T"] * _indeterminacy(y) + ax * (1 - ay) * _indeterminacy(x) * y["T"]) / den
    return _transfer(z, k, "T")


def calibrated_or(x, y, ax: float, ay: float, disj=tinf_or) -> Tuple:
    """Base-rate-calibrated disjunction: `disj` plus the transfer
    K = [ax (1-ay) F_x I_y + (1-ax) ay I_x F_y] / (ax + ay - ax ay) from indeterminacy to F. On normalised
    inputs its coarsening is SL comultiplication with base rates ax, ay. The base rate of the result is
    ax + ay - ax * ay."""
    z = disj(x, y)
    den = ax + ay - ax * ay
    if den <= 0:
        return z
    k = (ax * (1 - ay) * x["F"] * _indeterminacy(y) + (1 - ax) * ay * _indeterminacy(x) * y["F"]) / den
    return _transfer(z, k, "F")
