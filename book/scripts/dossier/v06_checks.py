"""Checks added in v0.6.

1. Theorem 7(c): the I component of the Frechet-type N-norm can lie above or below the
   I of the credal Frechet conjunction on the classical frame (two explicit normalised pairs).
2. Counterexample C1: delta = 1/15 recomputed by linear programming.
Writes results/v06_checks.json and asserts every claim.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS  # noqa: E402


def frechet_type_I(x1, x2):
    return max(x1[1], x2[1])


def credal_frechet_I(x1, x2):
    l1, u1 = x1[0], 1 - x1[2]
    l2, u2 = x2[0], 1 - x2[2]
    return min(u1, u2) - max(0.0, l1 + l2 - 1)


out = {}
pos = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
neg = ((0.5, 0.0, 0.5), (0.5, 0.0, 0.5))
for name, (a, b) in (("positive_example", pos), ("negative_example", neg)):
    assert abs(sum(a) - 1) < 1e-12 and abs(sum(b) - 1) < 1e-12
    d = frechet_type_I(a, b) - credal_frechet_I(a, b)
    out[name] = {"x1": a, "x2": b, "I_Ntype": frechet_type_I(a, b),
                 "I_credal": credal_frechet_I(a, b), "difference": d}
assert out["positive_example"]["difference"] == 1.0
assert out["negative_example"]["difference"] == -0.5

# C1: n = 3, T({x}) = 0.4 for singletons, T({y,z}) = 0.3 for pairs (F({x}) = 0.3, F(pair) = 0.4).
# delta = min eps s.t. exists P with T(A) - eps <= P(A) <= 1 - F(A) + eps for all proper A.
from scipy.optimize import linprog  # noqa: E402
import itertools  # noqa: E402
import numpy as np  # noqa: E402

n = 3
A_ub, b_ub = [], []
for k in (1, 2):
    for A in itertools.combinations(range(n), k):
        ind = np.zeros(n + 1)
        ind[list(A)] = 1.0
        T = 0.4 if k == 1 else 0.3
        F = 0.3 if k == 1 else 0.4
        # -P(A) - eps <= -T ;  P(A) - eps <= 1 - F
        r = -ind.copy(); r[n] = -1.0; A_ub.append(r); b_ub.append(-T)
        r = ind.copy(); r[n] = -1.0; A_ub.append(r); b_ub.append(1 - F)
A_eq = [np.r_[np.ones(n), 0.0]]
res = linprog(c=np.r_[np.zeros(n), 1.0], A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq),
              b_eq=[1.0], bounds=[(0, 1)] * n + [(0, None)], method="highs")
assert res.status == 0
out["C1_delta"] = res.fun
assert abs(res.fun - 1 / 15) < 1e-9

with open(os.path.join(RESULTS, "v06_checks.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("V06 CHECKS PASSED")
