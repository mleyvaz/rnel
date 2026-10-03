"""P1 (proved) + Florentin problem F2 (open): indeterminacy profiles.

For a coherent normalized dual NP, I(A) = U(A) - L(A). Proved: I(A) = I(not A) and
I(A u B) <= I(A) + I(B) for disjoint A, B. For belief functions I is a hypergraph cut
function: I(A) = sum of m(B) over focal sets B that straddle A.
Checks: (i) subadditivity on random coherent envelopes, n = 3, 4, 5;
(ii) n = 3: every symmetric profile satisfying the triangle inequalities is a cut profile;
(iii) n = 4, 5: is every coherent profile a hypergraph cut profile? (search; LP per case)
"""
import itertools

import numpy as np
from scipy.optimize import linprog

from common import events, complement, lower_envelope, save

rng = np.random.default_rng(5)
out = {}


def profile_from_L(L, n):
    return {A: 1 - L[A] - L[complement(A, n)] for A in L}


def is_cut_profile(I, n):
    """LP: w_B >= 0 for |B| >= 2, sum w <= 1, I(A) = sum_{B straddles A} w_B."""
    Bs = [frozenset(c) for k in range(2, n + 1) for c in itertools.combinations(range(n), k)]
    evs = list(I)
    A_eq = np.array([[1.0 if (B & A and not B <= A) else 0.0 for B in Bs] for A in evs])
    b_eq = np.array([I[A] for A in evs])
    r = linprog(np.zeros(len(Bs)), A_ub=np.ones((1, len(Bs))), b_ub=[1.0], A_eq=A_eq, b_eq=b_eq,
                bounds=[(0, None)] * len(Bs), method="highs")
    return r.status == 0


res = {}
for n in (3, 4, 5):
    viol_sub, noncut, cases, submod = 0, 0, 0, 0
    example = None
    for _ in range(1500 if n < 5 else 600):
        k = int(rng.integers(2, 6))
        d = rng.dirichlet(np.ones(n) * rng.uniform(0.2, 2), size=k)
        L = lower_envelope(d, n, events(n))
        I = profile_from_L(L, n)
        for A in I:
            for B in I:
                if not (A & B) and (A | B) in I and I[A | B] > I[A] + I[B] + 1e-9:
                    viol_sub += 1
        Ie = dict(I); Ie[frozenset()] = 0.0; Ie[frozenset(range(n))] = 0.0
        submod += all(Ie[A | B] + Ie[A & B] <= Ie[A] + Ie[B] + 1e-9 for A in Ie for B in Ie)
        if not is_cut_profile(I, n):
            noncut += 1
            if example is None:
                example = {"dists": d.tolist(), "I": {str(sorted(A)): v for A, v in I.items()}}
        cases += 1
    res[n] = {"cases": cases, "subadditivity_violations": viol_sub, "non_cut_profiles": noncut, "submodular_profiles": submod,
              "first_non_cut_example": example}
    assert viol_sub == 0
out["coherent_profiles"] = res

# n = 3: triangle region == cut profiles
tri_bad = 0
for _ in range(3000):
    i = rng.uniform(0, 1, 3)
    tri = all(i[k] <= i[(k + 1) % 3] + i[(k + 2) % 3] + 1e-12 for k in range(3))
    I = {}
    for x in range(3):
        I[frozenset({x})] = i[x]
        I[complement(frozenset({x}), 3)] = i[x]
    tri_bad += tri != is_cut_profile(I, 3)
out["n3_triangle_equals_cut"] = tri_bad == 0
assert tri_bad == 0

save("f2_indeterminacy_profiles.json", out)
print("F2 OK")
for n, v in res.items():
    print(" n=%d cases=%d subadd_viol=%d non_cut=%d submodular=%d" % (
        n, v["cases"], v["subadditivity_violations"], v["non_cut_profiles"], v["submodular_profiles"]))
print(" n=3 triangle == cut:", out["n3_triangle_equals_cut"])
