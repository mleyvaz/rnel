"""Checks added for manuscript v0.2 (fact corrections after the Codex veracity review).

1. Theorem 5(d): betting interval [T, 1-F] vs evidential interval [b, b+u] with
   (b, d, u) = rho(T, F) (saturating chart). Counterexample (0.2, 0.5, 0.3); the two
   intervals coincide iff T*F = 0; otherwise [T, 1-F] is strictly inside [b, b+u].
2. Theorem 9(c): event-wise comonotone model on the 16-atom product of two minimal
   liftings, M_co = {Q : marginals in K(x1), K(x2), and for X in {T,I,F}
   Q(E_X^A cap E_X^B) = min(Q(E_X^A), Q(E_X^B))}. M_co is a union of 8 polytopes
   (one per choice of the smaller marginal for each X); its lower envelope on the
   conjunction events is computed by 8 x 3 linear programmes and compared with
   the min/max/max N-norm.
3. n = 3: every coherent lower probability on all events is 2-monotone (proof in the
   paper); numerical check on random lower envelopes.
4. Number of pattern sets examined by the minimality search of t7_t8 for m = 3.
5. Which scripts contain assert statements; f1_f3_probes only flags F1.
6. Corollary 2 on the trivial events (IDM formulas give U(empty) != 0, L(Omega) != 1).
"""
import itertools
import os
import re

import numpy as np
from scipy.optimize import linprog

from common import save

rng = np.random.default_rng(29)
out = {}
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- 1. Theorem 5(d) -------------
def rho(T, F):
    den = 1 - T * F
    return T * (1 - F) / den, F * (1 - T) / den, (1 - T) * (1 - F) / den


b, d, u = rho(0.2, 0.3)
ce = {"triple": [0.2, 0.5, 0.3], "betting": [0.2, 0.7], "evidential": [b, b + u]}
assert abs(b - 0.148936) < 1e-6 and abs(b + u - 0.744681) < 1e-6
out["T5d_counterexample"] = ce

t5 = {"cases": 0, "coincide_when_TF_zero": 0, "strict_when_TF_pos": 0, "violations": 0}
grid = [(T, 0.0) for T in np.linspace(0, 0.99, 50)] + [(0.0, F) for F in np.linspace(0, 0.99, 50)]
for T, F in grid:
    b, d, u = rho(T, F)
    t5["cases"] += 1
    if abs(b - T) < 1e-12 and abs((b + u) - (1 - F)) < 1e-12:
        t5["coincide_when_TF_zero"] += 1
    else:
        t5["violations"] += 1
for T, F in rng.uniform(0, 1, (20000, 2)):
    b, d, u = rho(T, F)
    t5["cases"] += 1
    # b < T and b + u > 1 - F whenever T*F > 0
    if b < T - 1e-15 and (b + u) > (1 - F) + 1e-15:
        t5["strict_when_TF_pos"] += 1
    else:
        t5["violations"] += 1
assert t5["violations"] == 0
out["T5d_condition"] = t5

# ---------------------------------------------------------------- 2. Theorem 9(c) -------------
PATS = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
E = np.array([[p[k] for p in PATS] for k in range(3)], float)  # rows E_T, E_I, E_F
idx = [(i, j) for i in range(4) for j in range(4)]
MA = np.array([[E[k][i] for (i, j) in idx] for k in range(3)])  # Q(E_k^A)
MB = np.array([[E[k][j] for (i, j) in idx] for k in range(3)])  # Q(E_k^B)
# Q(E_k^A \ E_k^B) and Q(E_k^B \ E_k^A)
DAB = np.array([[E[k][i] * (1 - E[k][j]) for (i, j) in idx] for k in range(3)])
DBA = np.array([[E[k][j] * (1 - E[k][i]) for (i, j) in idx] for k in range(3)])
cT = np.array([E[0][i] * E[0][j] for (i, j) in idx])
cI = np.array([max(E[1][i], E[1][j]) for (i, j) in idx])
cF = np.array([max(E[2][i], E[2][j]) for (i, j) in idx])


def comon_envelope(x1, x2):
    best = [np.inf] * 3
    feasible = 0
    for choice in itertools.product((0, 1), repeat=3):
        # choice[k] = 0: E_k^A is (a.s.) inside E_k^B, so Q(cap) = Q(E_k^A) = min
        Aeq = [np.ones(16)]
        beq = [1.0]
        for k, ch in enumerate(choice):
            Aeq.append(DAB[k] if ch == 0 else DBA[k])
            beq.append(0.0)
        A_ub = np.vstack([-MA, -MB]); b_ub = np.r_[-x1, -x2]
        any_ok = False
        for c_i, c in enumerate((cT, cI, cF)):
            r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=np.array(Aeq), b_eq=beq,
                        bounds=[(0, None)] * 16, method="highs")
            if r.status == 0:
                any_ok = True
                best[c_i] = min(best[c_i], r.fun)
        feasible += any_ok
    return np.array(best), feasible


t9 = {"cases": 0, "max_err": 0.0, "min_feasible_polytopes": 8}
for _ in range(1500):
    x1, x2 = rng.uniform(0, 1, 3), rng.uniform(0, 1, 3)
    env, feas = comon_envelope(x1, x2)
    svns = np.array([min(x1[0], x2[0]), max(x1[1], x2[1]), max(x1[2], x2[2])])
    t9["max_err"] = max(t9["max_err"], float(np.max(np.abs(env - svns))))
    t9["min_feasible_polytopes"] = min(t9["min_feasible_polytopes"], feas)
    t9["cases"] += 1
assert t9["max_err"] < 1e-8 and t9["min_feasible_polytopes"] >= 1
out["T9c_eventwise_comonotone_LP"] = t9

# ---------------------------------------------------------------- 3. n = 3, 2-monotone --------
EV3 = [frozenset(c) for k in range(0, 4) for c in itertools.combinations(range(3), k)]


def env3(dists):
    return {A: (float(min(dd[list(A)].sum() for dd in dists)) if A else 0.0) for A in EV3}


def is_2mono(L):
    return all(L[A | B] + L[A & B] >= L[A] + L[B] - 1e-12 for A in EV3 for B in EV3)


n3 = {"envelopes": 0, "not_2monotone": 0}
for k in (2, 3, 4, 5, 6, 8):
    for _ in range(5000):
        dd = rng.dirichlet(np.ones(3) * rng.choice([0.3, 0.7, 1.0, 3.0]), size=k)
        n3["envelopes"] += 1
        n3["not_2monotone"] += (not is_2mono(env3(dd)))
assert n3["not_2monotone"] == 0
out["n3_coherent_are_2monotone_check"] = n3

# ---------------------------------------------------------------- 4. pattern sets (t7_t8) -----
# Same search as t7_t8_lifting_operators.py, with a counter. Same seed and draws.
rng3 = np.random.default_rng(3)


def lp_min(c, A_ub, b_ub, A_eq, b_eq, n):
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n,
                method="highs")
    return r.fun if r.status == 0 else None


def faithful(patterns, m, pts):
    na = len(patterns)
    Emat = np.array([[p[k] for p in patterns] for k in range(m)], float)
    for x in pts:
        for k in range(m):
            v = lp_min(Emat[k], -Emat, -np.asarray(x), np.ones((1, na)), [1.0], na)
            if v is None or abs(v - x[k]) > 1e-8:
                return False
    return True


for m in (3, 4, 5):  # consume the same random draws as t7_t8
    list(rng3.uniform(0, 1, (400, m)))
m = 3
allp = list(itertools.product((0, 1), repeat=m))
pts = list(itertools.product((0.0, 1.0), repeat=m)) + list(rng3.uniform(0, 1, (60, m)))
examined, minimal = 0, []
for size in range(1, 9):
    for S in itertools.combinations(allp, size):
        examined += 1
        if faithful(list(S), m, pts):
            minimal.append(S)
    if minimal:
        break
out["t8_pattern_sets_examined_m3"] = examined
out["t8_pattern_sets_nonempty_total_m3"] = 2 ** 8 - 1
out["t8_minimal_sets_m3"] = [list(S) for S in minimal]
assert examined == 8 + 28 + 56 + 70 == 162 and len(minimal) == 1

# ---------------------------------------------------------------- 5. asserts per script -------
asserts = {}
for f in ["common.py", "t1_reduction_hierarchy.py", "t2_intervals_idm.py", "t3_t4_t5_readings.py",
          "t7_t8_lifting_operators.py", "f2_indeterminacy_profiles.py", "f1_f3_probes.py"]:
    src = open(os.path.join(HERE, f), encoding="utf-8").read()
    asserts[f] = len(re.findall(r"^\s*assert\s", src, flags=re.M))
out["assert_statements_per_script"] = asserts
out["f1_f3_note"] = ("f1_f3_probes.py asserts only the F3 region check (Theorem 8(d)); "
                     "for F1 it records F1_conjecture_consistent without stopping")

# ---------------------------------------------------------------- 6. Corollary 2 trivial events
N, s = 10.0, 2.0
out["Cor2_trivial_events"] = {"U_empty_by_formula": s / (N + s), "L_Omega_by_formula": N / (N + s)}

save("v02_corrections.json", out)
print("v0.2 checks OK")
for k, v in out.items():
    print(" ", k, v)
