"""Evidence for Florentin problems F1 and F3 (numerical, not proofs).

F1: copula-induced conjunctions. On the lifted frame a copula C gives
    N_C = (C(T1,T2), I1+I2-C(I1,I2), F1+F2-C(F1,F2))  (verified for C = Pi, W, M in t7_t8).
    On the classical frame the credal conjunction under C is
    (C(T1,T2), C(1-F1,1-F2) - C(T1,T2), 1 - C(1-F1,1-F2)).
    Surplus sigma_C = I(N_C) - I(credal_C) on normalized inputs. For which C is sigma_C >= 0?
F3: faithful regions of sub-maximal liftings (disjoint frame; Belnap frame t, f, b, n).
"""
import numpy as np
from scipy.optimize import linprog

from common import save

rng = np.random.default_rng(13)
out = {}

# ---------------- F1: surplus sign by copula family ----------------------------------------
fam = {
    "Pi": lambda u, v: u * v,
    "M": lambda u, v: min(u, v),
    "W": lambda u, v: max(0.0, u + v - 1),
}
for th in (-1.0, -0.5, 0.5, 1.0):
    fam["FGM(%.1f)" % th] = (lambda t: (lambda u, v: u * v * (1 + t * (1 - u) * (1 - v))))(th)
for th in (0.5, 2.0, 8.0):
    fam["Clayton(%.1f)" % th] = (lambda t: (lambda u, v: 0.0 if min(u, v) == 0 else
                                            max(u ** -t + v ** -t - 1, 0) ** (-1 / t)))(th)
for th in (-5.0, -1.0, 1.0, 5.0):
    fam["Frank(%.1f)" % th] = (lambda t: (lambda u, v: -1 / t * np.log(
        1 + (np.exp(-t * u) - 1) * (np.exp(-t * v) - 1) / (np.exp(-t) - 1))))(th)

def _surv(C):
    return lambda u, v: u + v - 1 + C(1 - u, 1 - v)


def _gumbel(t):
    return lambda u, v: 0.0 if min(u, v) == 0 else np.exp(-((-np.log(u)) ** t + (-np.log(v)) ** t) ** (1 / t))


def _plackett(t):
    def C(u, v):
        s = 1 + (t - 1) * (u + v)
        return (s - np.sqrt(s * s - 4 * u * v * t * (t - 1))) / (2 * (t - 1))
    return C


fam["survClayton(2.0)"] = _surv(fam["Clayton(2.0)"])
fam["Gumbel(2.0)"] = _gumbel(2.0)
fam["survGumbel(2.0)"] = _surv(_gumbel(2.0))
fam["Plackett(3.0)"] = _plackett(3.0)
fam["Plackett(0.3)"] = _plackett(0.3)
RADIALLY_SYMMETRIC = {"Pi", "M", "W", "FGM", "Frank", "Plackett"}

X = rng.dirichlet([1, 1, 1], size=(20000, 2))
surplus = {}
for name, C in fam.items():
    s_min = np.inf
    for (T1, I1, F1), (T2, I2, F2) in X:
        I_N = I1 + I2 - C(I1, I2)
        I_c = C(1 - F1, 1 - F2) - C(T1, T2)
        s_min = min(s_min, I_N - I_c)
    surplus[name] = float(s_min)
out["F1_min_surplus_by_copula"] = surplus
for name, v in surplus.items():
    sym = name.split("(")[0] in RADIALLY_SYMMETRIC
    out.setdefault("F1_conjecture_consistent", True)
    if sym != (v >= -1e-9):
        out["F1_conjecture_consistent"] = False


# ---------------- F3: faithful regions ---------------------------------------------------------
def faithful_point(E, x):
    na = E.shape[1]
    for k in range(3):
        r = linprog(E[k], A_ub=-E, b_ub=-x, A_eq=np.ones((1, na)), b_eq=[1.0],
                    bounds=[(0, None)] * na, method="highs")
        if r.status != 0 or abs(r.fun - x[k]) > 1e-8:
            return False
    return True


# atoms as columns; rows E_T, E_I, E_F
disjoint = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)          # a, iota, not-a
belnap = np.array([[1, 0, 1, 0], [0, 0, 0, 1], [0, 1, 1, 0]], float)   # t, f, b, n
pts = rng.uniform(0, 1, (6000, 3))
mm = {"disjoint": 0, "belnap": 0}
for x in pts:
    T, I, F = x
    mm["disjoint"] += faithful_point(disjoint, x) != (T + I + F <= 1)
    mm["belnap"] += faithful_point(belnap, x) != (T + I <= 1 and F + I <= 1)
out["F3_region_mismatches"] = mm
assert mm["disjoint"] == 0 and mm["belnap"] == 0

save("f1_f3_probes.json", out)
print("F1 min surplus by copula:", {k: round(v, 4) for k, v in surplus.items()})
print("F3 region mismatches (disjoint: T+I+F<=1; Belnap: T+I<=1 and F+I<=1):", mm)
