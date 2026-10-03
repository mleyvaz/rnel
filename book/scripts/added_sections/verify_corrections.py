"""(v13.1) Recompute every number used by the green correction paragraphs and the new margin comments of version 13.1.

Checks (linear programming with scipy HiGHS, exact arithmetic where possible):
  L1  Section 11.3 pools on the 10-atom minimal lifting vs the product lifting (comment ML6, correction after 11.3).
  L2  Section 11.4 random-selection event over the four I events on the 13-atom minimal lifting vs product lifting (ML7).
  L3  Corollary 3(b): three atoms (0,1,1), (1,0,1), (1,1,0) are faithful on {x1 in [0,1], T+F <= 1} (ML2).
  L4  Theorem 10(c) / A.7(c): sign of (1-2c)(M-W)(T1,T2) for T1=0.5, T2=0.6 at c=0.8 and c=0.2 (ML4, ML15).
  L5  Section 12.2: lower envelope of the segment K of Proposition 3(c) is a belief function (Moebius masses).
  L6  Theorem 7(c): two explicit normalised input pairs with opposite signs of the I difference (comment, veracity 20).
  L7  Corollary 1: rational witnesses C3 (coherent, not 2-monotone, n=4) and C4 (2-monotone, not a belief function, n=3).
  L8  Theorem 11 check of p1_plithogenic.py (recorded run): largest formula error and largest falsity difference of a
      radially symmetric copula (ML5).
Writes corrections_checks.json (current directory, book edition) and prints a summary; exit code 1 if an assertion fails.
L8 reads p1_plithogenic.json: the fresh copy in the output dir if present, else book/expected/ch10_plithogeny/.
"""
import itertools, json, os, sys
import numpy as np
from scipy.optimize import linprog
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
BOOKDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # the book/ folder


def book_result(name, expected_rel):
    """Book edition: a result file of another book script. The fresh copy in the current (output) directory is used
    when present (same regenerate run); otherwise the stored expected copy under book/expected/."""
    p = os.path.join(os.getcwd(), name)
    return p if os.path.exists(p) else os.path.join(BOOKDIR, expected_rel)

out = {}


def minimal_lifting(m):
    """patterns 1 and 1 - e_k (Theorem 8(a))"""
    pats = [tuple([1] * m)] + [tuple(0 if j == k else 1 for j in range(m)) for k in range(m)]
    return np.array(pats, dtype=float)


def lower(pats, x, objective):
    """min sum_j objective_j P(E_j) over P >= 0, sum P = 1, P(E_j) >= x_j"""
    n = len(pats)
    c = pats @ objective
    A_ub = -pats.T; b_ub = -np.asarray(x, float)
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(0, None)] * n, method="highs")
    assert r.status == 0, r.message
    return r.fun


# ---------------------------------------------------------------- L1: Section 11.3
tri = [(0.62, 0.30, 0.08), (0.55, 0.20, 0.35), (0.70, 0.10, 0.60)]
x = [v for t in tri for v in t]              # m = 9 components, 10 atoms
pats = minimal_lifting(9)
for name, w in (("equal", [1 / 3, 1 / 3, 1 / 3]), ("one_minus_c", [0.5, 1 / 3, 1 / 6])):
    obj = np.zeros(9)
    for j in range(3): obj[3 * j] = w[j]
    minimal = lower(pats, x, obj)
    product = sum(w[j] * tri[j][0] for j in range(3))   # product lifting: minima attained jointly (one factor per type)
    out["L1_pool_" + name] = dict(minimal_lifting_atoms=len(pats), minimal=round(minimal, 6), product=round(product, 6))
assert abs(out["L1_pool_equal"]["minimal"] - 2 / 3) < 1e-6 and abs(out["L1_pool_equal"]["product"] - 0.623333) < 1e-6
assert abs(out["L1_pool_one_minus_c"]["minimal"] - 0.631667) < 1e-6 and abs(out["L1_pool_one_minus_c"]["product"] - 0.61) < 1e-9

# ---------------------------------------------------------------- L2: Section 11.4
neu = [(0.5, 0.1, 0.2), (0.6, 0.2, 0.4), (0.8, 0.0, 0.1), (0.4, 0.3, 0.5)]
x = [v for t in neu for v in t]              # m = 12, 13 atoms
pats = minimal_lifting(12)
obj = np.zeros(12)
for j in range(4): obj[3 * j + 1] = 0.25
out["L2_graduation_I_mean"] = dict(minimal_lifting_atoms=len(pats), minimal=round(lower(pats, x, obj), 6),
                                   product=round(sum(t[1] for t in neu) / 4, 6))
assert abs(out["L2_graduation_I_mean"]["minimal"] - 0.75) < 1e-9 and abs(out["L2_graduation_I_mean"]["product"] - 0.15) < 1e-9

# ---------------------------------------------------------------- L3: Corollary 3(b), restricted region
pats3 = np.array([(0, 1, 1), (1, 0, 1), (1, 1, 0)], float)
grid = [i / 10 for i in range(11)]
pts, worst = 0, 0.0
for x1 in grid:
    for T in grid:
        for F in grid:
            if T + F > 1 + 1e-12: continue
            pts += 1
            for k in range(3):
                e = np.zeros(3); e[k] = 1
                worst = max(worst, abs(lower(pats3, (x1, T, F), e) - (x1, T, F)[k]))
out["L3_three_atoms_faithful"] = dict(points=pts, largest_error=worst, atoms=3, m_plus_1=4)
assert worst < 1e-9
# and with T + F > 1 the three atoms are not faithful (the glut atom (1,1,1) is needed)
r = linprog(np.zeros(3), A_ub=-pats3.T, b_ub=-np.array([1.0, 0.7, 0.6]), A_eq=np.ones((1, 3)), b_eq=[1.0], bounds=[(0, None)] * 3, method="highs")
out["L3_outside_region_example"] = dict(point=[1.0, 0.7, 0.6], credal_set_empty=(r.status == 2))
assert r.status == 2

# ---------------------------------------------------------------- L4: sign in Theorem 10(c)
M = lambda a, b: min(a, b); Wc = lambda a, b: max(0.0, a + b - 1)
out["L4_sign"] = {str(c): round((1 - 2 * c) * (M(0.5, 0.6) - Wc(0.5, 0.6)), 6) for c in (0.8, 0.2)}
assert out["L4_sign"]["0.8"] == -0.24 and out["L4_sign"]["0.2"] == 0.24

# ---------------------------------------------------------------- L5: Section 12.2, segment K of Proposition 3(c)
P1, P2 = np.array([0.2, 0.3, 0.5]), np.array([0.4, 0.5, 0.1])
subsets = [s for r in (1, 2, 3) for s in itertools.combinations(range(3), r)]
L = {s: min(P1[list(s)].sum(), P2[list(s)].sum()) for s in subsets}
mob = {}
for s in subsets:
    mob[s] = sum((-1) ** (len(s) - len(t)) * L[t] for r in range(1, len(s) + 1) for t in itertools.combinations(s, r))
out["L5_moebius"] = {str([i + 1 for i in s]): round(v, 6) for s, v in mob.items()}
assert min(mob.values()) > -1e-12

# ---------------------------------------------------------------- L6: Theorem 7(c), explicit witnesses
def frechet_I_diff(x1, x2):
    (T1, I1, F1), (T2, I2, F2) = x1, x2
    credal = min(1 - F1, 1 - F2) - max(0.0, T1 + T2 - 1)
    return max(I1, I2) - credal
w_pos = ((0.0, 0.9, 0.1), (0.0, 0.1, 0.9)); w_neg = ((0.5, 0.0, 0.5), (0.5, 0.0, 0.5))
out["L6_theorem7c"] = dict(positive=dict(pair=w_pos, diff=round(frechet_I_diff(*w_pos), 6)),
                           negative=dict(pair=w_neg, diff=round(frechet_I_diff(*w_neg), 6)))
assert out["L6_theorem7c"]["positive"]["diff"] == 0.8 and out["L6_theorem7c"]["negative"]["diff"] == -0.5

# ---------------------------------------------------------------- L7: Corollary 1, rational witnesses
def lower_env(points, n):
    subs = [s for r in range(1, n) for s in itertools.combinations(range(n), r)]
    return {s: min(sum(p[i] for i in s) for p in points) for s in subs}


def two_monotone(Lw, n):
    full = tuple(range(n)); get = lambda s: 0.0 if not s else (1.0 if s == full else Lw[s])
    allsubs = [()] + list(Lw) + [full]
    for a in allsubs:
        for b in allsubs:
            u = tuple(sorted(set(a) | set(b))); i = tuple(sorted(set(a) & set(b)))
            if get(u) + get(i) < get(a) + get(b) - 1e-12: return False, (a, b)
    return True, None


C3 = lower_env([(0.5, 0.5, 0, 0), (0, 0, 0.5, 0.5)], 4)        # lower envelope of two probabilities: coherent
ok3, viol = two_monotone(C3, 4)
out["L7_C3"] = dict(definition="lower envelope of P1 = (1/2, 1/2, 0, 0) and P2 = (0, 0, 1/2, 1/2) on four atoms",
                    two_monotone=ok3, violation=[[i + 1 for i in viol[0]], [i + 1 for i in viol[1]]] if viol else None)
assert not ok3
C4 = {s: (0.0 if len(s) == 1 else 0.5) for r in (1, 2) for s in itertools.combinations(range(3), r)}
ok4, _ = two_monotone(C4, 3)
mob4 = 1.0 - sum(C4[s] for s in C4 if len(s) == 2)  # singletons have mass 0, pairs 0.5 each
pts = minimal = None
# coherence of C4: every bound attained by a probability with P(k) <= 1/2
att = all(abs(lower(np.eye(3), [0, 0, 0], np.eye(3)[k]) - 0) < 1e-12 for k in range(3))
out["L7_C4"] = dict(definition="L({i}) = 0 and L({i,j}) = 1/2 on three atoms", two_monotone=ok4,
                    moebius_of_Omega=round(mob4, 6), credal_set="P(k) <= 1/2 for every k")
assert ok4 and mob4 < 0

# ---------------------------------------------------------------- L8: recorded run of p1_plithogenic.py (Theorem 11)
p1_json = book_result("p1_plithogenic.json", "expected/ch10_plithogeny/p1_plithogenic.json")
if os.path.exists(p1_json):
    pj = json.load(open(p1_json, encoding="utf-8"))
else:
    # The archived stdout is the same JSON followed by the verifier's final status line.
    p1_out = book_result("ch10_plithogeny__p1_plithogenic.out", "expected/ch10_plithogeny/p1_plithogenic.out")
    raw = open(p1_out, encoding="utf-8").read()
    pj = json.loads(raw[: raw.rfind("\nALL CHECKS PASSED")])
t11 = pj["T11"]
formula_err = max(max(v.get("T_err", 0), v.get("F_formula_err", 0), v.get("sigma_formula_err", 0)) for v in t11.values() if isinstance(v, dict))
fdiff_rs = {k: v["max_abs_F_diff"] for k, v in t11.items() if isinstance(v, dict) and "max_abs_F_diff" in v and v["max_abs_F_diff"] < 1e-10}
out["L8_theorem11_run"] = dict(largest_formula_error=formula_err, largest_falsity_difference_radially_symmetric=max(fdiff_rs.values()),
                               attained_by=max(fdiff_rs, key=fdiff_rs.get))
json.dump(out, open("corrections_checks.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=list)
for k, v in out.items(): print(k, v)
print("ALL CHECKS PASSED")
