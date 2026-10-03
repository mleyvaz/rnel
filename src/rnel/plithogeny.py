"""rnel.plithogeny (experimental): Smarandache's plithogenic probability and N-norms read as credal objects.

Book reference: Chapter 10 ("Plithogenic probability and credal sets", Sections 10.1-10.5) and the applications of
Chapter 11. The statements implemented and tested here are:

  Corollary 3   plithogenic lifting: m + 1 atoms (m = sum of the components of all attribute values) represent a
                plithogenic probability faithfully and coherently; the product lifting has prod(m_k + 1) atoms.
  Theorem 10    plithogenic N-norms are natural extensions of a conjunction switched by the contradiction degree c:
                (a) for every copula C the lower envelope is N^pl_{C,c}; (b) with no dependence assumption, W on truth
                and M on falsity when c <= 1/2, M on truth and W on falsity when c >= 1/2; (c) at c = 1/2 every model
                gives the componentwise mean, and only there; (d) the indeterminacy (I1 + I2)/2 never depends on the
                dependence model.
  Theorem 11    plithogenic N-norms on the classical frame: truth exact; falsity gap (1 - 2c)(Ĉ(F1,F2) - C(F1,F2)), so
                falsity is exact iff c = 1/2 or C is radially symmetric; closed form of the indeterminacy surplus.
  Corollary 4   plithogenic IDM with indeterminate observations.
  Corollary 5   the three neutrosophic conjunctions compared by Smarandache are switched conjunctions with switch vector
                (0, gamma_I, 1), gamma_I = 0, 1, 1/2; only gamma_I = 1/2 is dependence-free in I.
  Proposition 2 many attribute values: the binary rule is commutative, not associative (defect c(1-c)(d - a) for Pi);
                n-ary switched conjunctions; the mean is the order-free and dependence-free aggregate.

A *switch* with probability g turns a conjunction into a disjunction. For two events with lower probabilities a and b
and a copula C, the probability of the g-switched conjunction is

    g (a + b) + (1 - 2g) C(a, b)          (Definition 5 with r = C(a, b))

and with no dependence assumption its lower envelope uses W when g <= 1/2 and M when g >= 1/2 (Theorem 10(b)).
The plithogenic N-norm is the switched N-norm with switch vector (c, 1/2, 1 - c) on (T, I, F).

Core functions are pure Python; the linear-programming verifiers (``*_lp``, ``is_faithful``) need scipy + numpy.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .copula import Copula, lower_frechet, pi_copula, survival, upper_frechet

Triple = Tuple[float, float, float]

__all__ = [
    # Definition 5 / Theorem 10
    "switched_probability", "switched_lower", "switched_nnorm", "plith_nnorm", "plith_nnorm_nodep",
    "plith_nnorm_lp", "is_dependence_free",
    # Theorem 11
    "classical_frame",
    # Corollary 3
    "minimal_lifting", "product_lifting", "lifting_atom_counts", "is_faithful",
    # Corollary 4
    "plithogenic_idm",
    # Corollary 5
    "SMARANDACHE_CONJUNCTIONS", "smarandache_conjunction",
    # Proposition 2
    "switch_truth", "associativity_defect", "nary_switched", "nary_switched_lp", "selection_mean",
    # applications
    "pcm_step", "pcm_aggregate", "contradiction_pool",
]


def _check_unit(name: str, v: float) -> None:
    if not (-1e-12 <= v <= 1 + 1e-12):
        raise ValueError(f"{name} must lie in [0, 1], got {v}")


# ============================================================================ Definition 5 / Theorem 10
def switched_probability(p: float, q: float, r: float, g: float, mode: str = "conj") -> float:
    """Probability of the g-switched conjunction (mode "conj") or disjunction (mode "disj") of two events with
    Q(A) = p, Q(B) = q and Q(A & B) = r (Definition 5): conj = (1-g) r + g (p + q - r); disj exchanges r and p+q-r."""
    if mode == "conj":
        return (1 - g) * r + g * (p + q - r)
    if mode == "disj":
        return (1 - g) * (p + q - r) + g * r
    raise ValueError("mode must be 'conj' or 'disj'")


def switched_lower(a: float, b: float, g: float, C: Optional[Copula] = None) -> float:
    """Lower envelope (natural extension) of the g-switched conjunction of two events with lower probabilities a, b.

    With a copula C (event-wise C-coupling, Theorem 10(a)): g (a + b) + (1 - 2g) C(a, b).
    With ``C=None`` (no dependence assumption, Theorem 10(b)): W if g <= 1/2, M if g >= 1/2.
    """
    _check_unit("switch", g)
    if C is None:
        C = lower_frechet if g <= 0.5 else upper_frechet
    return g * (a + b) + (1 - 2 * g) * C(a, b)


def switched_nnorm(x: Sequence[float], y: Sequence[float], switches: Sequence[float],
                   C: Optional[Copula] = None) -> Triple:
    """N-norm with one switch per component: (switched_lower(T1,T2,gT), switched_lower(I1,I2,gI),
    switched_lower(F1,F2,gF)). ``C=None`` means no dependence assumption."""
    gT, gI, gF = switches
    return (switched_lower(x[0], y[0], gT, C), switched_lower(x[1], y[1], gI, C), switched_lower(x[2], y[2], gF, C))


def plith_nnorm(x: Sequence[float], y: Sequence[float], c: float, C: Copula = pi_copula) -> Triple:
    """Plithogenic N-norm N^pl_{C,c}(x, y) for a copula C and contradiction degree c (Theorem 10(a)):

        T = (1-c) C(T1,T2) + c (T1 + T2 - C(T1,T2))
        I = (I1 + I2) / 2                       (= the 1/2-switch, for every C: Theorem 10(d))
        F = (1-c) (F1 + F2 - C(F1,F2)) + c C(F1,F2)

    Under strong independence (C = Pi) this is the natural extension on the product of two minimal liftings.
    """
    _check_unit("c", c)
    return switched_nnorm(x, y, (c, 0.5, 1 - c), C)


def plith_nnorm_nodep(x: Sequence[float], y: Sequence[float], c: float) -> Triple:
    """Natural extension of the plithogenic N-norm with no dependence assumption (Theorem 10(b)):
    W on truth and M on falsity when c <= 1/2; M on truth and W on falsity when c >= 1/2; I = (I1 + I2)/2."""
    _check_unit("c", c)
    return switched_nnorm(x, y, (c, 0.5, 1 - c), None)


# minimal lifting of one triple (m = 3): patterns of the atoms in the order (T, I, F)
_PAT3 = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]


def plith_nnorm_lp(x: Sequence[float], y: Sequence[float], c: float, gamma_I: float = 0.5) -> Triple:
    """Natural extension by linear programming on the 16 atoms of the product of two minimal liftings, with no
    dependence assumption: min over joints whose marginals lie in K(x), K(y) of each switched event probability,
    switch vector (c, gamma_I, 1 - c). Verifier of Theorem 10(b) and Corollary 5 (needs scipy)."""
    import numpy as np
    from scipy.optimize import linprog

    atoms = list(itertools.product(range(4), range(4)))
    switches = (c, gamma_I, 1 - c)
    A_ub, b_ub = [], []
    for comp in range(3):
        A_ub.append([-_PAT3[i][comp] for (i, j) in atoms]); b_ub.append(-float(x[comp]))
        A_ub.append([-_PAT3[j][comp] for (i, j) in atoms]); b_ub.append(-float(y[comp]))
    out = []
    for X, g in enumerate(switches):
        obj = np.zeros(16)
        for k, (i, j) in enumerate(atoms):
            a, b = _PAT3[i][X], _PAT3[j][X]
            obj[k] = (1 - g) * a * b + g * max(a, b)
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=[np.ones(16)], b_eq=[1], bounds=(0, 1), method="highs")
        if res.status != 0:
            raise RuntimeError(f"linprog failed: {res.message}")
        out.append(float(res.fun))
    return tuple(out)


def is_dependence_free(c: float, copulas: Sequence[Copula] = (lower_frechet, pi_copula, upper_frechet),
                       x: Sequence[float] = (0.3, 0.2, 0.6), y: Sequence[float] = (0.7, 0.5, 0.1),
                       tol: float = 1e-12) -> bool:
    """True when N^pl_{C,c}(x, y) is the same for all the given copulas and for the no-assumption extension.
    By Theorem 10(c) this holds for all inputs iff c = 1/2."""
    vals = [plith_nnorm(x, y, c, C) for C in copulas] + [plith_nnorm_nodep(x, y, c)]
    return all(max(abs(a - b) for a, b in zip(v, vals[0])) <= tol for v in vals[1:])


# ============================================================================ Theorem 11
def classical_frame(x: Sequence[float], y: Sequence[float], c: float, C: Copula = pi_copula) -> Dict[str, float]:
    """Plithogenic N-norm against the credal conjunction on the classical frame (Theorem 11), normalised inputs.

    Returns L and U (credal lower/upper probability of the c-switched conjunction under C, with P(A) in [T1, 1-F1],
    P(B) in [T2, 1-F2]), the plithogenic triple, and
      truth_gap      T_pl - L                                      (= 0, part (a))
      falsity_gap    F_pl - (1 - U)  = (1 - 2c)(Ĉ(F1,F2) - C(F1,F2))   (part (b))
      surplus        I_pl - (U - L)  = (1 - 2c)((I1+I2)/2 - (C(u1,u2) - C(T1,T2)))   (part (c))
    together with the closed forms ``falsity_gap_formula`` and ``surplus_formula``.
    """
    (T1, I1, F1), (T2, I2, F2) = x, y
    u1, u2 = 1 - F1, 1 - F2
    L = c * (T1 + T2) + (1 - 2 * c) * C(T1, T2)
    U = c * (u1 + u2) + (1 - 2 * c) * C(u1, u2)
    pl = plith_nnorm(x, y, c, C)
    Ch = survival(C)
    return dict(
        L=L, U=U, T=pl[0], I=pl[1], F=pl[2],
        truth_gap=pl[0] - L,
        falsity_gap=pl[2] - (1 - U),
        falsity_gap_formula=(1 - 2 * c) * (Ch(F1, F2) - C(F1, F2)),
        surplus=pl[1] - (U - L),
        surplus_formula=(1 - 2 * c) * ((I1 + I2) / 2 - (C(u1, u2) - C(T1, T2))),
    )


# ============================================================================ Corollary 3
def minimal_lifting(ms: Sequence[int] | int) -> List[Tuple[int, ...]]:
    """Atoms of the minimal lifting for attribute values with m_1, ..., m_n components (Corollary 3(a)):
    m + 1 atoms with m = sum m_k; atom k (k < m) lies in every component event except the k-th, the last atom lies
    in all of them. Each atom is returned as its 0/1 membership pattern over the m component events."""
    m = ms if isinstance(ms, int) else int(sum(ms))
    if m < 1:
        raise ValueError("need at least one component")
    atoms = [tuple(0 if j == k else 1 for j in range(m)) for k in range(m)]
    atoms.append(tuple([1] * m))
    return atoms


def product_lifting(ms: Sequence[int]) -> List[Tuple[Tuple[int, ...], ...]]:
    """Atoms of the product of the n minimal liftings of the attribute values (Corollary 3(c)): prod(m_k + 1)
    atoms, each a tuple of per-value patterns. Faithful and coherent, not minimal when n >= 2."""
    return list(itertools.product(*[minimal_lifting(m) for m in ms]))


def lifting_atom_counts(ms: Sequence[int]) -> Tuple[int, int]:
    """(atoms of the minimal lifting, atoms of the product lifting) = (sum m_k + 1, prod (m_k + 1))."""
    return int(sum(ms)) + 1, int(math.prod(m + 1 for m in ms))


def is_faithful(x: Sequence[float], ms: Optional[Sequence[int]] = None, product: bool = False,
                tol: float = 1e-9) -> bool:
    """Check by linear programming that the lifting represents the component vector ``x`` faithfully: the lower
    envelope of the credal set {Q : Q(E_k) >= x_k} on each component event E_k equals x_k (Corollary 3).
    ``ms`` splits x into attribute values (default: one value with len(x) components). Needs scipy."""
    import numpy as np
    from scipy.optimize import linprog

    x = [float(v) for v in x]
    ms = [len(x)] if ms is None else list(ms)
    if sum(ms) != len(x):
        raise ValueError("sum(ms) must equal len(x)")
    if product:
        atoms = product_lifting(ms)
        A = np.array([[a[v][k] for a in atoms] for v, m in enumerate(ms) for k in range(m)], dtype=float)
    else:
        A = np.array(minimal_lifting(ms), dtype=float).T
    nat = A.shape[1]
    for k in range(len(x)):
        r = linprog(A[k], A_ub=-A, b_ub=-np.array(x), A_eq=[np.ones(nat)], b_eq=[1], bounds=(0, 1), method="highs")
        if r.status != 0 or abs(r.fun - x[k]) > tol:
            return False
    return True


# ============================================================================ Corollary 4
def plithogenic_idm(counts: Sequence[float], n_indeterminate: float = 0.0, s: float = 2.0,
                    event: Optional[Sequence[int]] = None):
    """Plithogenic imprecise Dirichlet model for one attribute value (Corollary 4).

    With counts n_i on k >= 2 categories, n? indeterminate (unclassifiable) observations, N = sum n_i + n? and s > 0,
    the triple of category i is T_i = n_i/(N+s), I_i = (n? + s)/(N+s), F_i = 1 - T_i - I_i; it is coherent and its
    natural extension to an event A is [n_A/(N+s), (n_A + n? + s)/(N+s)]. Returns the list of triples, or the
    triple of ``event`` (indices of categories) when given.
    """
    if s <= 0:
        raise ValueError("s must be positive")
    n = [float(v) for v in counts]
    if len(n) < 2:
        raise ValueError("need k >= 2 categories")
    N = sum(n) + n_indeterminate
    den = N + s
    I = (n_indeterminate + s) / den
    if event is not None:
        nA = sum(n[i] for i in event)
        if len(set(event)) == len(n):
            return (1.0, 0.0, 0.0)
        T = nA / den
        return (T, I, 1 - T - I)
    return [(ni / den, I, 1 - ni / den - I) for ni in n]


# ============================================================================ Corollary 5
#: switch on indeterminacy of the three conjunctions of Smarandache (2019b, eqs. 171, 174, 178)
SMARANDACHE_CONJUNCTIONS = {"eq171": 0.0, "eq174": 1.0, "plithogenic": 0.5}


def smarandache_conjunction(x: Sequence[float], y: Sequence[float], kind: str | float = "plithogenic",
                            C: Optional[Copula] = None) -> Triple:
    """The three neutrosophic conjunctions compared by Smarandache as switched conjunctions with switch vector
    (0, gamma_I, 1) (Corollary 5): ``kind`` is "eq171" (gamma_I = 0), "eq174" (gamma_I = 1), "plithogenic"
    (gamma_I = 1/2) or a number. With ``C=None`` (no dependence assumption) the indeterminacy is
    max(0, I1+I2-1), max(I1, I2) and (I1+I2)/2 respectively."""
    g = SMARANDACHE_CONJUNCTIONS[kind] if isinstance(kind, str) else float(kind)
    return switched_nnorm(x, y, (0.0, g, 1.0), C)


# ============================================================================ Proposition 2
def switch_truth(a: float, b: float, c: float, C: Copula = pi_copula) -> float:
    """f_c(a, b) = c(a + b) + (1 - 2c) C(a, b): the truth component of N^pl_{C,c} (Proposition 2)."""
    return c * (a + b) + (1 - 2 * c) * C(a, b)


def associativity_defect(a: float, b: float, d: float, c: float, C: Copula = pi_copula) -> float:
    """f_c(f_c(a, b), d) - f_c(a, f_c(b, d)). For C = Pi it equals c(1 - c)(d - a) (Proposition 2(a))."""
    return switch_truth(switch_truth(a, b, c, C), d, c, C) - switch_truth(a, switch_truth(b, d, c, C), c, C)


def nary_switched(xs: Sequence[float], c: float, model: str = "independence") -> float:
    """Natural extension of the n-ary c-switched conjunction (1-c) Q(and A_i) + c Q(or A_i) (Proposition 2(b)):
    independence (1-c) prod x_i + c (1 - prod(1 - x_i)); comonotone (1-c) min x_i + c max x_i."""
    xs = [float(v) for v in xs]
    if model == "independence":
        return (1 - c) * math.prod(xs) + c * (1 - math.prod(1 - v for v in xs))
    if model == "comonotone":
        return (1 - c) * min(xs) + c * max(xs)
    raise ValueError("model must be 'independence' or 'comonotone'")


def nary_switched_lp(xs: Sequence[float], c: float) -> Tuple[float, float]:
    """(min, max) of the n-ary c-switched conjunction over all couplings with marginals x_i (no dependence
    assumption), by LP over the 2^n atoms. For n >= 3 the spread at c = 1/2 is positive (Proposition 2(c))."""
    import numpy as np
    from scipy.optimize import linprog

    n = len(xs)
    pats = list(itertools.product([0, 1], repeat=n))
    obj = np.array([(1 - c) * all(s) + c * any(s) for s in pats], dtype=float)
    Aeq = [[s[i] for s in pats] for i in range(n)] + [[1] * len(pats)]
    beq = [float(v) for v in xs] + [1.0]
    lo = linprog(obj, A_eq=Aeq, b_eq=beq, bounds=(0, 1), method="highs").fun
    hi = -linprog(-obj, A_eq=Aeq, b_eq=beq, bounds=(0, 1), method="highs").fun
    return float(lo), float(hi)


def selection_mean(xs: Sequence[float]) -> float:
    """Natural extension of the random-selection event A_J (J uniform, independent of the frame): the mean
    (1/n) sum x_i under every dependence model (Proposition 2(c)); equal to the 1/2-switch for n = 2."""
    xs = [float(v) for v in xs]
    return sum(xs) / len(xs)


# ============================================================================ applications (Chapter 11)
def pcm_step(state: float, weight: float, c: float) -> float:
    """One edge of a plithogenic cognitive map (Martin and Smarandache 2020): the c-switched product conjunction
    of 'the source is active' and 'the edge transmits' under strong independence, f_c(state, weight) with C = Pi."""
    return switch_truth(state, weight, c, pi_copula)


def pcm_aggregate(influences: Sequence[float]) -> float:
    """Combination of the incoming influences of a concept: the maximum, which is the lower probability of their
    union with no dependence assumption (Fréchet bounds)."""
    return max(influences)


def contradiction_pool(triples: Sequence[Sequence[float]], contradictions: Sequence[float]) -> Triple:
    """Pool assessments of several attribute values with weights w_k proportional to 1 - c_k (c_k = contradiction
    degree to the dominant value), as in application A3 of Chapter 11. With all c_k equal this is the mean."""
    w = [1 - c for c in contradictions]
    tot = sum(w)
    if tot <= 0:
        raise ValueError("all contradiction degrees equal 1")
    return tuple(sum(wk * t[i] for wk, t in zip(w, triples)) / tot for i in range(3))
