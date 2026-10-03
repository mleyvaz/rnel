"""Step 4. N-norms and plithogeny on the product lifting: the triple of 'Jenifer graduates'.

(a) Table 8 of the book (classical frame, four forms, four dependence models), recomputed.
(b) The graduation triple on the product lifting (256 atoms): E_T = all T events, E_I = some I event,
    E_F = some F event (Theorem 9 extended to four conjuncts), under no assumption (LP), strong
    independence (product of extreme points), full comonotonicity and comonotonicity within subject.
(c) Plithogenic indeterminacy (c_{T,I} = 1/2): the 1/2-switched four-fold conjunction of the I events
    and the random-selection mean (Proposition 2(b), (c)).
"""
import itertools

import numpy as np

from re_common import (COURSES, NEUTRO, SUBJECT, intervals, frechet_lp, lp_min, product_lifting,
                       product_constraints, vertices_K)

FORMS = ["fuzzy", "interval", "intuitionistic", "neutrosophic"]


def indep(iv):
    return np.prod([iv[c][0] for c in COURSES]), np.prod([iv[c][1] for c in COURSES])


def comon(iv):
    return min(iv[c][0] for c in COURSES), min(iv[c][1] for c in COURSES)


def comon_in_subject(iv):
    lo = min(iv["DE"][0], iv["SA"][0]) * min(iv["FM"][0], iv["SM"][0])
    hi = min(iv["DE"][1], iv["SA"][1]) * min(iv["FM"][1], iv["SM"][1])
    return lo, hi


def fmt(p):
    lo, hi = p
    return f"{lo:.3f}" if abs(hi - lo) < 1e-12 else f"[{lo:.3f}, {hi:.3f}]"


print("Step 4a: Table 8 recomputed (probability that Jenifer graduates, classical frame)")
print(f"{'model':18}" + "".join(f"{f:>18}" for f in FORMS))
book = {  # values printed in Table 8 of v10
    "indep.": ["0.096", "[0.019, 0.189]", "[0.096, 0.216]", "[0.096, 0.216]"],
    "comon. in subject": ["0.200", "[0.060, 0.300]", "[0.200, 0.300]", "[0.200, 0.300]"],
    "comonotone": ["0.400", "[0.200, 0.500]", "[0.400, 0.500]", "[0.400, 0.500]"],
    "no assump. (LP)": ["[0.000, 0.400]", "[0.000, 0.500]", "[0.000, 0.500]", "[0.000, 0.500]"],
}
rows = {"indep.": indep, "comon. in subject": comon_in_subject, "comonotone": comon,
        "no assump. (LP)": frechet_lp}
for name, f in rows.items():
    vals = [fmt(f(intervals(form))) for form in FORMS]
    print(f"{name:18}" + "".join(f"{v:>18}" for v in vals))
    for v, w in zip(vals, book[name]):
        assert v == w.replace("[0, ", "[0.000, "), (name, v, w)
print("  all 16 cells agree with Table 8 of the book")

# ------------------------------------------------------------------ (b) graduation triple, product lifting
n, E = product_lifting()
A, b = product_constraints(E, NEUTRO)
EG = {"T": np.prod([E[c]["T"] for c in COURSES], axis=0),
      "I": np.max([E[c]["I"] for c in COURSES], axis=0),
      "F": np.max([E[c]["F"] for c in COURSES], axis=0)}
noas = tuple(lp_min(EG[X], A, b, n) for X in "TIF")

V = {c: vertices_K(NEUTRO[c]) for c in COURSES}
pe = {c: {X: V[c] @ np.array([1.0 if p[j] else 0.0 for p in
                              [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]])
          for j, X in enumerate("TIF")} for c in COURSES}
best = {"T": np.inf, "I": np.inf, "F": np.inf, "half": np.inf}
for idx in itertools.product(*[range(len(V[c])) for c in COURSES]):
    pT = [pe[c]["T"][i] for c, i in zip(COURSES, idx)]
    pI = [pe[c]["I"][i] for c, i in zip(COURSES, idx)]
    pF = [pe[c]["F"][i] for c, i in zip(COURSES, idx)]
    qT, qIu, qIn, qF = np.prod(pT), 1 - np.prod(1 - np.array(pI)), np.prod(pI), 1 - np.prod(1 - np.array(pF))
    best["T"] = min(best["T"], qT); best["I"] = min(best["I"], qIu); best["F"] = min(best["F"], qF)
    best["half"] = min(best["half"], 0.5 * qIn + 0.5 * qIu)
ind = (best["T"], best["I"], best["F"])

T_, I_, F_ = ({c: NEUTRO[c][j] for c in COURSES} for j in range(3))
com = (min(T_.values()), max(I_.values()), max(F_.values()))
subj = {s: [c for c in COURSES if SUBJECT[c] == s] for s in ("math", "mech")}
sub = (np.prod([min(T_[c] for c in subj[s]) for s in subj]),
       1 - np.prod([1 - max(I_[c] for c in subj[s]) for s in subj]),
       1 - np.prod([1 - max(F_[c] for c in subj[s]) for s in subj]))

print("\nStep 4b: triple of 'graduates' on the product lifting (T of all, I of some, F of some)")
for name, t in [("no assumption (LP, 256 atoms)", noas), ("strong independence (vertices)", ind),
                ("comonotone (closed form)", com), ("comonotone within subject", sub)]:
    print(f"  {name:32} ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")
assert np.allclose(noas, (0.0, 0.3, 0.5), atol=1e-9)
assert np.allclose(ind, (0.096, 0.496, 0.784), atol=1e-9)

# ------------------------------------------------------------------ (c) plithogenic indeterminacy
EIn = np.prod([E[c]["I"] for c in COURSES], axis=0)
half = 0.5 * EIn + 0.5 * EG["I"]
half_noas = lp_min(half, A, b, n)
mean_I = sum(E[c]["I"] for c in COURSES) / 4
mean_noas = lp_min(mean_I, A, b, n)
half_com = 0.5 * min(I_.values()) + 0.5 * max(I_.values())
print("\nStep 4c: plithogenic indeterminacy of graduation (switch 1/2 on the I events)")
print(f"  1/2-switch, strong independence  = {best['half']:.3f}")
print(f"  1/2-switch, comonotone           = {half_com:.3f}")
print(f"  1/2-switch, no assumption (LP)   = {half_noas:.3f}")
print(f"  random-selection mean, any model = {mean_noas:.3f}")
assert abs(best["half"] - 0.248) < 1e-9 and abs(half_com - 0.15) < 1e-12 and abs(mean_noas - 0.15) < 1e-9
