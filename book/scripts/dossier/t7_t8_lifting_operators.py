"""T7 (credal conjunctions on the classical frame vs neutrosophic N-norms: exact surplus of I)
T8 (glut lifting: faithful coherent representation of the whole cube on m+1 atoms; minimality)
T9 (on the lifted frame the three N-norm families ARE credal natural extensions).
"""
import itertools

import numpy as np
from scipy.optimize import linprog

from common import save

rng = np.random.default_rng(3)
out = {}


def lp_min(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, n=None):
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n,
                method="highs")
    return r.fun if r.status == 0 else None


# ---------------------------------------------------------------- T8: lifting ------------------
def lifted_lower(patterns, x):
    """patterns: list of 0/1 tuples (atoms); x: lower bounds for events E_k = {atoms with k-th bit 1}.
    Returns lower envelope of each E_k over K = {P : P(E_k) >= x_k}, or None if K empty."""
    na, m = len(patterns), len(x)
    Emat = np.array([[p[k] for p in patterns] for k in range(m)], float)
    res = []
    for k in range(m):
        v = lp_min(Emat[k], A_ub=-Emat, b_ub=-np.asarray(x), A_eq=np.ones((1, na)), b_eq=[1.0], n=na)
        if v is None:
            return None
        res.append(v)
    return np.array(res)


def faithful(patterns, m, pts):
    for x in pts:
        lo = lifted_lower(patterns, x)
        if lo is None or np.max(np.abs(lo - x)) > 1e-8:
            return False
    return True


lift = {}
for m in (3, 4, 5):
    pats = [tuple(1 - int(j == k) for j in range(m)) for k in range(m)] + [tuple([1] * m)]
    pts = list(itertools.product((0.0, 1.0), repeat=m)) + list(rng.uniform(0, 1, (400, m)))
    lift[m] = {"atoms": pats, "faithful": faithful(pats, m, pts)}
    assert lift[m]["faithful"]
# minimality for m = 3 by exhaustive enumeration of atom sets (2^8 subsets)
m = 3
allp = list(itertools.product((0, 1), repeat=m))
pts = list(itertools.product((0.0, 1.0), repeat=m)) + list(rng.uniform(0, 1, (60, m)))
minimal = []
for size in range(1, 9):
    for S in itertools.combinations(allp, size):
        if faithful(list(S), m, pts):
            minimal.append(S)
    if minimal:
        break
out["lifting"] = {str(k): v for k, v in lift.items()}
out["lifting_min_size_m3"] = len(minimal[0])
out["lifting_minimal_sets_m3"] = [list(S) for S in minimal]
assert len(minimal) == 1 and set(minimal[0]) == {(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)}

# ---------------------------------------------------------------- T7: classical frame ----------
def rand_norm():
    return rng.dirichlet([1, 1, 1])


t7 = {"cases": 0, "frechet_err": 0.0, "indep_err": 0.0, "alg_surplus_err": 0.0,
      "svns_surplus_min": 1.0, "luk_I_diff_min": 1.0, "luk_I_diff_max": -1.0}
for _ in range(3000):
    (T1, I1, F1), (T2, I2, F2) = rand_norm(), rand_norm()
    l1, u1, l2, u2 = T1, 1 - F1, T2, 1 - F2
    # Frechet: joint q on atoms (AB, A~B, ~AB, ~A~B) with P(A) in [l1,u1], P(B) in [l2,u2]
    PA, PB = np.array([1, 1, 0, 0.]), np.array([1, 0, 1, 0.])
    A_ub = np.array([-PA, PA, -PB, PB]); b_ub = np.array([-l1, u1, -l2, u2])
    AB = np.array([1, 0, 0, 0.])
    lo = lp_min(AB, A_ub, b_ub, np.ones((1, 4)), [1.0], 4)
    hi = -lp_min(-AB, A_ub, b_ub, np.ones((1, 4)), [1.0], 4)
    t7["frechet_err"] = max(t7["frechet_err"], abs(lo - max(0, l1 + l2 - 1)), abs(hi - min(u1, u2)))
    # strong independence: bilinear, extremes at interval endpoints
    prods = [x * y for x in (l1, u1) for y in (l2, u2)]
    g = [x * y for x in np.linspace(l1, u1, 7) for y in np.linspace(l2, u2, 7)]
    t7["indep_err"] = max(t7["indep_err"], abs(min(g) - l1 * l2), abs(max(g) - u1 * u2),
                          abs(min(prods) - l1 * l2))
    # credal-product triple vs algebraic N-norm
    Tc, Fc = T1 * T2, 1 - (1 - F1) * (1 - F2)
    Ic = 1 - Tc - Fc
    Ta, Ia, Fa = T1 * T2, I1 + I2 - I1 * I2, F1 + F2 - F1 * F2
    assert abs(Ta - Tc) < 1e-12 and abs(Fa - Fc) < 1e-12
    t7["alg_surplus_err"] = max(t7["alg_surplus_err"], abs((Ia - Ic) - (I1 * F2 + I2 * F1)))
    # comonotone credal vs SVNS (min, max, max)
    Tm, Um = min(T1, T2), min(u1, u2)
    assert abs(Tm - min(T1, T2)) < 1e-12 and abs((1 - Um) - max(F1, F2)) < 1e-12
    t7["svns_surplus_min"] = min(t7["svns_surplus_min"], max(I1, I2) - (Um - Tm))
    # Frechet credal vs Lukasiewicz-type N-norm (max(0,T1+T2-1), max I, max F)
    If = min(u1, u2) - max(0, T1 + T2 - 1)
    d = max(I1, I2) - If
    t7["luk_I_diff_min"] = min(t7["luk_I_diff_min"], d)
    t7["luk_I_diff_max"] = max(t7["luk_I_diff_max"], d)
    t7["cases"] += 1
assert t7["frechet_err"] < 1e-8 and t7["indep_err"] < 1e-12 and t7["alg_surplus_err"] < 1e-12
assert t7["svns_surplus_min"] >= -1e-12
out["T7_classical_frame"] = t7

# ---------------------------------------------------------------- T9: lifted frame -------------
PATS = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
E = np.array([[p[k] for p in PATS] for k in range(3)], float)  # rows: E_T, E_I, E_F


def vertices(x):
    """Vertices of K(x) = {p in simplex(4) : E p >= x} by active-set enumeration."""
    G = np.vstack([np.eye(4), E]); h = np.r_[np.zeros(4), x]  # G p >= h
    V = []
    for S in itertools.combinations(range(7), 3):
        Aeq = np.vstack([G[list(S)], np.ones(4)]); beq = np.r_[h[list(S)], 1.0]
        if abs(np.linalg.det(Aeq)) < 1e-12:
            continue
        p = np.linalg.solve(Aeq, beq)
        if np.all(G @ p >= h - 1e-10):
            V.append(p)
    return V


t9 = {"cases": 0, "indep_err": 0.0, "frechet_err": 0.0, "comon_err": 0.0}
for _ in range(1500):
    x1, x2 = rng.uniform(0, 1, 3), rng.uniform(0, 1, 3)  # arbitrary triples, off-normalization
    V1, V2 = vertices(x1), vertices(x2)
    # strong independence: product measures of vertices (bilinear objective)
    PT = [(E[0] @ p) * (E[0] @ q) for p in V1 for q in V2]
    PI = [1 - (1 - E[1] @ p) * (1 - E[1] @ q) for p in V1 for q in V2]
    PF = [1 - (1 - E[2] @ p) * (1 - E[2] @ q) for p in V1 for q in V2]
    alg = (x1[0] * x2[0], x1[1] + x2[1] - x1[1] * x2[1], x1[2] + x2[2] - x1[2] * x2[2])
    t9["indep_err"] = max(t9["indep_err"], abs(min(PT) - alg[0]), abs(min(PI) - alg[1]),
                          abs(min(PF) - alg[2]))
    # comonotone (pairwise maximal dependence): P(E and E') = min, P(E or E') = max
    CT = [min(E[0] @ p, E[0] @ q) for p in V1 for q in V2]
    CI = [max(E[1] @ p, E[1] @ q) for p in V1 for q in V2]
    CF = [max(E[2] @ p, E[2] @ q) for p in V1 for q in V2]
    svns = (min(x1[0], x2[0]), max(x1[1], x2[1]), max(x1[2], x2[2]))
    t9["comon_err"] = max(t9["comon_err"], abs(min(CT) - svns[0]), abs(min(CI) - svns[1]),
                          abs(min(CF) - svns[2]))
    # Frechet: LP over joint Q on 16 atoms with marginals in K(x1), K(x2)
    idx = [(i, j) for i in range(4) for j in range(4)]
    M1 = np.array([[E[k][i] for (i, j) in idx] for k in range(3)])
    M2 = np.array([[E[k][j] for (i, j) in idx] for k in range(3)])
    A_ub = np.vstack([-M1, -M2]); b_ub = np.r_[-x1, -x2]
    cT = np.array([E[0][i] * E[0][j] for (i, j) in idx])
    cI = np.array([max(E[1][i], E[1][j]) for (i, j) in idx])
    cF = np.array([max(E[2][i], E[2][j]) for (i, j) in idx])
    fr = [lp_min(c, A_ub, b_ub, np.ones((1, 16)), [1.0], 16) for c in (cT, cI, cF)]
    luk = (max(0, x1[0] + x2[0] - 1), max(x1[1], x2[1]), max(x1[2], x2[2]))
    t9["frechet_err"] = max(t9["frechet_err"], *[abs(a - b) for a, b in zip(fr, luk)])
    t9["cases"] += 1
assert t9["indep_err"] < 1e-9 and t9["comon_err"] < 1e-9 and t9["frechet_err"] < 1e-8
out["T9_lifted_frame"] = t9

save("t7_t8_lifting_operators.json", out)
print("T7/T8/T9 OK")
print(" lifting m=3,4,5 faithful:", {k: v["faithful"] for k, v in lift.items()},
      "; minimal size m=3:", out["lifting_min_size_m3"], minimal[0])
print(" T7:", t7)
print(" T9:", t9)
