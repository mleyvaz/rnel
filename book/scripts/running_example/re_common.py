"""Shared data and helpers for the running example (Jenifer's graduation, Smarandache 2017, pp. 135-137).

The rnel library is imported from PYTHONPATH (set by `python -m rnel.book regenerate`); the optional
environment variable RNEL_SRC can point to another source tree. All linear programmes use scipy's HiGHS solver.
"""
import itertools
import os
import sys

import numpy as np
from scipy.optimize import linprog

RNEL_SRC = os.environ.get("RNEL_SRC")
if RNEL_SRC and RNEL_SRC not in sys.path:
    sys.path.insert(0, RNEL_SRC)

from rnel import neutro_credal as nc  # noqa: E402

COURSES = ["DE", "SA", "FM", "SM"]  # differential equations, stochastic analysis, fluid, solid mechanics
SUBJECT = {"DE": "math", "SA": "math", "FM": "mech", "SM": "mech"}

# The four published forms of the adviser's plithogenic probability (Smarandache 2017, pp. 135-137)
FUZZY = {"DE": 0.5, "SA": 0.6, "FM": 0.8, "SM": 0.4}
INTERVAL = {"DE": (0.4, 0.6), "SA": (0.3, 0.7), "FM": (0.8, 0.9), "SM": (0.2, 0.5)}
INTUITIONISTIC = {"DE": (0.5, 0.2), "SA": (0.6, 0.4), "FM": (0.8, 0.1), "SM": (0.4, 0.5)}  # (pass, fail)
NEUTRO = {"DE": (0.5, 0.1, 0.2), "SA": (0.6, 0.2, 0.4), "FM": (0.8, 0.0, 0.1), "SM": (0.4, 0.3, 0.5)}

LP_OPTS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


def intervals(form):
    """Betting reading on the classical frame {pass, fail}: [l, u] per course."""
    if form == "fuzzy":
        return {k: (v, v) for k, v in FUZZY.items()}
    if form == "interval":
        return dict(INTERVAL)
    if form == "intuitionistic":
        return {k: (p, 1 - f) for k, (p, f) in INTUITIONISTIC.items()}
    if form == "neutrosophic":
        return {k: (t, 1 - f) for k, (t, i, f) in NEUTRO.items()}
    raise ValueError(form)


def lp_min(c, A_ub, b_ub, n):
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, n)), b_eq=[1.0],
                bounds=[(0, None)] * n, method="highs", options=LP_OPTS)
    assert r.status == 0, r.message
    return float(r.fun)


# ---------------------------------------------------------------- classical frame: 2^4 pass/fail atoms
CLASSICAL_ATOMS = np.array(list(itertools.product([0, 1], repeat=4)), dtype=float)  # 1 = pass


def frechet_lp(iv):
    """Lower and upper P(all four passed) over every joint law with the given marginal intervals."""
    n = len(CLASSICAL_ATOMS)
    A, b = [], []
    for k, c in enumerate(COURSES):
        l, u = iv[c]
        A.append(-CLASSICAL_ATOMS[:, k]); b.append(-l)
        A.append(CLASSICAL_ATOMS[:, k]); b.append(u)
    g = CLASSICAL_ATOMS.prod(axis=1)
    lo = lp_min(g, np.array(A), np.array(b), n)
    hi = -lp_min(-g, np.array(A), np.array(b), n)
    return lo, hi


# ---------------------------------------------------------------- glut liftings
MIN_PAT = np.array([(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)], dtype=float)  # (T, I, F) bits


def minimal_lifting_patterns(m):
    """Theorem 8(a) / Corollary 3(a): m + 1 atoms with patterns 1 - e_k and 1."""
    pats = [np.ones(m) - np.eye(m)[k] for k in range(m)] + [np.ones(m)]
    return np.array(pats)


def product_lifting():
    """Corollary 3(c): product of four 4-atom minimal liftings, 4^4 = 256 atoms.
    Returns E[c][X] indicators (course c, component X in 'TIF')."""
    atoms = list(itertools.product(range(4), repeat=4))
    E = {c: {X: np.array([MIN_PAT[a[k], j] for a in atoms]) for j, X in enumerate("TIF")}
         for k, c in enumerate(COURSES)}
    return len(atoms), E


def product_constraints(E, triples):
    A, b = [], []
    for c in COURSES:
        for j, X in enumerate("TIF"):
            A.append(-E[c][X]); b.append(-triples[c][j])
    return np.array(A), np.array(b)


def vertices_K(triple):
    """Extreme points of K(T, I, F) on the 4-atom minimal lifting (brute-force active sets)."""
    n = 4
    G = np.vstack([-MIN_PAT.T, -np.eye(n)])          # G p <= h
    h = np.concatenate([-np.asarray(triple, float), np.zeros(n)])
    verts = []
    for act in itertools.combinations(range(len(G)), n - 1):
        M = np.vstack([G[list(act)], np.ones(n)])
        rhs = np.concatenate([h[list(act)], [1.0]])
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        p = np.linalg.solve(M, rhs)
        if np.all(G @ p <= h + 1e-10) and not any(np.allclose(p, v) for v in verts):
            verts.append(p)
    return np.array(verts)
