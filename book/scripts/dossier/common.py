"""Shared utilities for the neutrosophic-probability / credal-set dossier.

Frame Omega = {0, ..., n-1}. Events are frozensets. A neutrosophic probability
assignment (NP) is a dict  A -> (T, I, F).  The betting ("Walley") reading used
throughout: T(A) is a lower probability of A and F(A) a lower probability of
not-A, so the credal set is  M(NP) = {P : T(A) <= P(A) <= 1 - F(A) for all A}.
All LPs use scipy's HiGHS solver.
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
from scipy.optimize import linprog

TOL = 1e-7
RESULTS = os.getcwd()  # book edition: outputs go to the current working directory


def events(n, proper=True):
    """All events of an n-element frame (proper: without empty set and Omega)."""
    out = []
    for k in range(0 if not proper else 1, n + (0 if proper else 1)):
        for c in itertools.combinations(range(n), k):
            out.append(frozenset(c))
    return out


def complement(A, n):
    return frozenset(range(n)) - A


def indicator(A, n):
    v = np.zeros(n)
    v[list(A)] = 1.0
    return v


def _constraints(np_assign, n, eps_var=False):
    """Rows for  P(A) >= T - eps  and  P(A) <= 1 - F + eps  in A_ub x <= b_ub form."""
    A_ub, b_ub = [], []
    for A, (T, I, F) in np_assign.items():
        v = indicator(A, n)
        if eps_var:
            A_ub.append(np.r_[-v, -1.0]); b_ub.append(-T)
            A_ub.append(np.r_[v, -1.0]); b_ub.append(1.0 - F)
        else:
            A_ub.append(-v); b_ub.append(-T)
            A_ub.append(v); b_ub.append(1.0 - F)
    return np.array(A_ub), np.array(b_ub)


def credal_nonempty(np_assign, n):
    A_ub, b_ub = _constraints(np_assign, n)
    r = linprog(np.zeros(n), A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, n)), b_eq=[1.0],
                bounds=[(0, None)] * n, method="highs")
    return r.status == 0


def sure_loss_degree(np_assign, n):
    """delta = min eps >= 0 such that {P : T-eps <= P(A) <= 1-F+eps} is non-empty."""
    A_ub, b_ub = _constraints(np_assign, n, eps_var=True)
    c = np.r_[np.zeros(n), 1.0]
    A_eq = np.r_[np.ones(n), 0.0].reshape(1, -1)
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                bounds=[(0, None)] * n + [(0, None)], method="highs")
    assert r.status == 0, r.message
    return r.fun


def natural_extension(np_assign, n, B, upper=False):
    """min (or max) P(B) over the credal set; None if the credal set is empty."""
    A_ub, b_ub = _constraints(np_assign, n)
    c = indicator(B, n) * (-1.0 if upper else 1.0)
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, n)), b_eq=[1.0],
                bounds=[(0, None)] * n, method="highs")
    if r.status != 0:
        return None
    return -r.fun if upper else r.fun


def is_coherent(np_assign, n, tol=1e-6):
    """Walley coherence of the pair (T, 1-F): every bound attained by the credal set."""
    if not credal_nonempty(np_assign, n):
        return False
    for A, (T, I, F) in np_assign.items():
        lo = natural_extension(np_assign, n, A)
        hi = natural_extension(np_assign, n, A, upper=True)
        if abs(lo - T) > tol or abs(hi - (1 - F)) > tol:
            return False
    return True


def lower_envelope(dists, n, evs):
    """Lower envelope of a finite set of distributions (rows of dists) on events."""
    return {A: float(min(d[list(A)].sum() for d in dists)) for A in evs}


def np_from_lower(L, n):
    """Dual normalized NP from a lower probability L on proper events:
    T = L(A), F = L(not A), I = 1 - T - F."""
    out = {}
    for A in L:
        T = L[A]
        F = L[complement(A, n)]
        out[A] = (T, 1.0 - T - F, F)
    return out


def mobius(L, n):
    """Mobius inverse of a set function given on all events incl. empty/Omega."""
    allev = [frozenset(c) for k in range(n + 1) for c in itertools.combinations(range(n), k)]
    m = {}
    for A in allev:
        s = 0.0
        for k in range(len(A) + 1):
            for B in itertools.combinations(sorted(A), k):
                B = frozenset(B)
                s += (-1) ** (len(A) - len(B)) * L[B]
        m[A] = s
    return m


def full_L(L, n):
    """Add L(empty)=0 and L(Omega)=1."""
    out = dict(L)
    out[frozenset()] = 0.0
    out[frozenset(range(n))] = 1.0
    return out


def is_2monotone(L, n, tol=1e-9):
    Lf = full_L(L, n)
    allev = list(Lf)
    for A in allev:
        for B in allev:
            if Lf[A | B] + Lf[A & B] < Lf[A] + Lf[B] - tol:
                return False
    return True


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=float)
