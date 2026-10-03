"""open_problems.py (book edition of v10_open_problems.py) -- checks added in v1.0: progress on the open problems F1 and F2 (Section 8.3).

F1. sigma_C(x1, x2) = I1 + I2 - C(I1, I2) - C(1 - F1, 1 - F2) + C(T1, T2) on normalised triples.
  1. Symbolic (sympy, C an arbitrary function): on the face F1 = F2 = 0, sigma_C = Chat - C, and
     D = C - Chat satisfies D(1-u, 1-v) = -D(u, v)  (Theorem 9).
  2. Symbolic: sigma_Pi = I1 F2 + I2 F1 under normalisation.
  3. Exact (Fractions): sigma_M >= 0 and sigma_W >= 0 on the 1/12 grid of pairs of normalised triples.
  4. Exact (Fractions): the shuffle of M with permutation (0, 2, 1, 3), built from its measure, is a
     copula (boundary, margins, 2-increasing on the 1/24 grid), radially symmetric and exchangeable on
     that grid, and sigma = -1/4 at x1 = x2 = (1/2, 1/4, 1/4); minimum over the 1/8 grid  (Proposition 2).
F2. Indeterminacy profiles I = U - L of credal sets.
  5. Proposition 3 (indicator inequalities) on random credal sets (n = 3..6) with random valid indicator identities.
  6. Corollary 3: the profile (0.4, 0.3, 0.4, 0.3, 0.4, 0.1, 0.1) checked exactly (Fractions) for symmetry,
     subadditivity on all disjoint pairs and the bound 1, and for the violated star inequality;
     cross-check of unattainability by linear programming; sanity check on random lower envelopes.
  7. Numerical observation: profiles satisfying the three conditions and the four star inequalities
     that the linear programme declares unattainable (count on a grid sample).
  8. Proposition 4: exact check of the example profile (three conditions and the four star inequalities
     hold, I({0,3}) != 1 although I({1}) = I({3}) = 1), LP cross-check, and the proposition itself on
     random credal sets that contain two Dirac measures.
  9. Exact certificates for the counterexamples C3 (coherent, not 2-monotone) and C4 (2-monotone, not a
     belief function) of Section 3.1.
Numbering in the paper (v1.0): Theorem 9; Propositions 2, 3 (indicator inequalities) and 4; Corollary 3.
Writes v10_open_problems.json to the current directory and asserts every claim.
Input: ../../data/inputs/t1_reduction_hierarchy.json (stored output of dossier/t1_reduction_hierarchy.py).
"""
import itertools
import json
import os
import sys
from fractions import Fraction as Fr

import numpy as np
import sympy as sp
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS  # noqa: E402

out = {}

# ---------------------------------------------------------------- 1-2 symbolic
u, v = sp.symbols("u v")
C = sp.Function("C")


def Chat(a, b):
    return a + b - 1 + C(1 - a, 1 - b)


D = lambda a, b: C(a, b) - Chat(a, b)  # noqa: E731
antisym = sp.simplify(D(1 - u, 1 - v) + D(u, v))
assert antisym == 0
I1, I2, T1, T2, F1, F2 = sp.symbols("I1 I2 T1 T2 F1 F2")


def sigma_sym(Cf, t1, i1, f1, t2, i2, f2):
    return i1 + i2 - Cf(i1, i2) - Cf(1 - f1, 1 - f2) + Cf(t1, t2)


# face F1 = F2 = 0, T_k = 1 - I_k; C(1, 1) = 1 for a copula
face = sigma_sym(C, 1 - I1, I1, 0, 1 - I2, I2, 0).subs(C(1, 1), 1)
assert sp.simplify(face - (Chat(I1, I2) - C(I1, I2))) == 0
Pi = lambda a, b: a * b  # noqa: E731
sig_pi = sigma_sym(Pi, 1 - I1 - F1, I1, F1, 1 - I2 - F2, I2, F2)
assert sp.expand(sig_pi - (I1 * F2 + I2 * F1)) == 0
out["F1_symbolic"] = {"D(1-u,1-v)+D(u,v)": str(antisym), "face_equals_Chat_minus_C": True,
                      "sigma_Pi_equals_I1F2_plus_I2F1": True}


# ---------------------------------------------------------------- 3 exact M, W
def simplex(k):
    return [(Fr(a, k), Fr(b, k), Fr(k - a - b, k)) for a in range(k + 1) for b in range(k + 1 - a)]


def sigma(Cf, x1, x2):
    (t1, i1, f1), (t2, i2, f2) = x1, x2
    return i1 + i2 - Cf(i1, i2) - Cf(1 - f1, 1 - f2) + Cf(t1, t2)


Mc = lambda a, b: min(a, b)  # noqa: E731
Wc = lambda a, b: max(Fr(0), a + b - 1)  # noqa: E731
Pc = lambda a, b: a * b  # noqa: E731
pts12 = simplex(12)
mins = {}
for name, Cf in (("M", Mc), ("W", Wc), ("Pi", Pc)):
    mins[name] = min(sigma(Cf, a, b) for a in pts12 for b in pts12)
    assert mins[name] >= 0
out["F1_exact_grid_1_12"] = {"pairs": len(pts12) ** 2, **{k: str(m) for k, m in mins.items()}}

# ---------------------------------------------------------------- 4 shuffle of M
PERM = (0, 2, 1, 3)
K = 4


def S(a, b):
    """Mass of the four diagonal segments inside [0,a] x [0,b]; cell i -> cell PERM[i].
    On cell i the segment is {(i/K + t/K, PERM[i]/K + t/K) : t in [0,1]} with mass 1/K."""
    a, b = Fr(a), Fr(b)
    tot = Fr(0)
    for i in range(K):
        j = PERM[i]
        ta = min(max(a * K - i, Fr(0)), Fr(1))
        tb = min(max(b * K - j, Fr(0)), Fr(1))
        tot += Fr(1, K) * min(ta, tb)
    return tot


g24 = [Fr(k, 24) for k in range(25)]
boundary = all(S(0, x) == 0 and S(x, 0) == 0 and S(1, x) == x and S(x, 1) == x for x in g24)
two_inc = all(S(b, d) - S(a, d) - S(b, c) + S(a, c) >= 0
              for a, b in itertools.combinations(g24, 2) for c, d in itertools.combinations(g24, 2))
radial = all(S(x, y) == x + y - 1 + S(1 - x, 1 - y) for x in g24 for y in g24)
exch = all(S(x, y) == S(y, x) for x in g24 for y in g24)
assert boundary and two_inc and radial and exch
x0 = (Fr(1, 2), Fr(1, 4), Fr(1, 4))
vals = {"S(1/4,1/4)": S(Fr(1, 4), Fr(1, 4)), "S(3/4,3/4)": S(Fr(3, 4), Fr(3, 4)),
        "S(1/2,1/2)": S(Fr(1, 2), Fr(1, 2))}
s0 = sigma(S, x0, x0)
assert vals == {"S(1/4,1/4)": Fr(1, 4), "S(3/4,3/4)": Fr(3, 4), "S(1/2,1/2)": Fr(1, 4)}
assert s0 == Fr(-1, 4)
pts8 = simplex(8)
smin = min(sigma(S, a, b) for a in pts8 for b in pts8)
assert smin == Fr(-1, 4)
# the shuffle is not M, Pi or W, and it differs from M (its value at (1/2,1/2) is 1/4, not 1/2)
out["F1_shuffle"] = {"boundary_and_margins_1_24": boundary, "two_increasing_1_24": two_inc,
                     "radially_symmetric_1_24": radial, "exchangeable_1_24": exch,
                     **{k: str(w) for k, w in vals.items()}, "sigma_at_(1/2,1/4,1/4)x2": str(s0),
                     "min_sigma_grid_1_8": str(smin), "grid_1_8_pairs": len(pts8) ** 2}

# ---------------------------------------------------------------- 5 Proposition 3 random
rng = np.random.default_rng(2026)


def allevents(n):
    return [frozenset(c) for k in range(n + 1) for c in itertools.combinations(range(n), k)]


def envelope(P, n):
    ev = allevents(n)
    lo = {A: float(min(p[list(A)].sum() for p in P)) for A in ev}
    hi = {A: float(max(p[list(A)].sum() for p in P)) for A in ev}
    return {A: hi[A] - lo[A] for A in ev}


def ind(A, n):
    x = np.zeros(n)
    x[list(A)] = 1
    return x


tests = 0
worst = np.inf
for trial in range(4000):
    n = int(rng.integers(3, 7))
    P = rng.dirichlet(np.ones(n) * float(rng.choice([0.3, 1, 3])), size=int(rng.integers(2, 7)))
    I = envelope(P, n)
    ev = [A for A in allevents(n) if A]
    k = int(rng.integers(1, 5))
    Xs = [ev[int(rng.integers(len(ev)))] for _ in range(k)]
    lam = rng.integers(1, 4, size=k).astype(float)
    f = sum(l * ind(X, n) for l, X in zip(lam, Xs))
    # write f = mu 1_A + kappa 1_Omega when f takes at most two values
    valsf = sorted(set(np.round(f, 12)))
    if len(valsf) > 2:
        continue
    if len(valsf) == 1:
        A, mu, kappa = frozenset(), 0.0, valsf[0]
    else:
        lo_, hi_ = valsf
        A = frozenset(int(w) for w in range(n) if abs(f[w] - hi_) < 1e-9)
        mu, kappa = hi_ - lo_, lo_
        if rng.random() < 0.5:   # same identity written with the complement and mu < 0
            A = frozenset(range(n)) - A
            mu, kappa = -mu, hi_
    assert np.allclose(f, mu * ind(A, n) + kappa)
    slack = sum(l * I[X] for l, X in zip(lam, Xs)) - abs(mu) * I[A]
    worst = min(worst, slack)
    tests += 1
assert worst >= -1e-12
out["Proposition3_random"] = {"identities_tested": tests, "min_slack": worst}

# ---------------------------------------------------------------- 6 Corollary 3
n = 4
full = frozenset(range(n))
EV = allevents(n)
proper = [A for A in EV if 0 < len(A) < n]


def make_profile(vals7):
    i = {frozenset(): Fr(0), full: Fr(0)}
    reps = [frozenset([0]), frozenset([1]), frozenset([2]), frozenset([3]),
            frozenset([0, 1]), frozenset([0, 2]), frozenset([0, 3])]
    for A, x in zip(reps, vals7):
        i[A] = x
        i[full - A] = x
    return i


def three_conditions(i):
    sym = all(i[A] == i[full - A] for A in EV)
    bound = all(0 <= i[A] <= 1 for A in EV)
    sub = all(i[A | B] <= i[A] + i[B] for A in EV for B in EV if not (A & B))
    return sym, sub, bound


def star_values(i):
    s = i[frozenset([0, 1])] + i[frozenset([0, 2])] + i[frozenset([0, 3])]
    return {a: s - 2 * i[frozenset([a])] for a in range(n)}


cex = make_profile([Fr(4, 10), Fr(3, 10), Fr(4, 10), Fr(3, 10), Fr(4, 10), Fr(1, 10), Fr(1, 10)])
sym, sub, bound = three_conditions(cex)
star = star_values(cex)
assert sym and sub and bound and star[0] == Fr(-2, 10)


def attainable(i, tol=0.0):
    """LP feasibility: L on all events, one P_A per event with P_A >= L and P_A(A) = L(A),
    L(A) + L(A^c) = 1 - i(A)."""
    nE = len(EV)
    idx = {A: k for k, A in enumerate(EV)}
    nv = nE + nE * n
    Pv = lambda A, w: nE + idx[A] * n + w  # noqa: E731
    Aub, bub, Aeq, beq = [], [], [], []
    r = np.zeros(nv); r[idx[frozenset()]] = 1; Aeq.append(r); beq.append(0)
    r = np.zeros(nv); r[idx[full]] = 1; Aeq.append(r); beq.append(1)
    for A in EV:
        r = np.zeros(nv)
        for w in range(n):
            r[Pv(A, w)] = 1
        Aeq.append(r); beq.append(1)
        for B in EV:
            r = np.zeros(nv)
            for w in B:
                r[Pv(A, w)] = -1
            r[idx[B]] += 1
            Aub.append(r); bub.append(0)
        r = np.zeros(nv)
        for w in A:
            r[Pv(A, w)] = 1
        r[idx[A]] -= 1
        Aeq.append(r); beq.append(0)
    for A in proper:
        r = np.zeros(nv); r[idx[A]] = 1; r[idx[full - A]] += 1
        Aeq.append(r); beq.append(1 - float(i[A]))
    res = linprog(np.zeros(nv), A_ub=np.array(Aub), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq,
                  bounds=[(0, 1)] * nv, method="highs")
    return res.status == 0


assert not attainable(cex)
sanity = 0
for t in range(200):
    P = rng.dirichlet(np.ones(n), size=int(rng.integers(2, 6)))
    L = {A: min(p[list(A)].sum() for p in P) for A in EV}
    i = {A: Fr(1) - Fr(L[A]) - Fr(L[full - A]) for A in EV}
    i[frozenset()] = Fr(0); i[full] = Fr(0)
    sanity += attainable(i)
assert sanity == 200
star_min = np.inf
for t in range(3000):
    P = rng.dirichlet(np.ones(n) * float(rng.choice([0.3, 1, 3])), size=int(rng.integers(2, 6)))
    I = envelope(P, n)
    s = I[frozenset([0, 1])] + I[frozenset([0, 2])] + I[frozenset([0, 3])]
    star_min = min(star_min, min(s - 2 * I[frozenset([a])] for a in range(n)))
assert star_min >= -1e-12
out["Corollary3"] = {"profile": ["0.4", "0.3", "0.4", "0.3", "0.4", "0.1", "0.1"],
                     "symmetry": sym, "subadditivity_all_disjoint_pairs": sub, "bound": bound,
                     "star_slack_by_a": {str(a): str(s) for a, s in star.items()},
                     "LP_attainable": False, "sanity_random_envelopes_attainable": f"{sanity}/200",
                     "star_on_3000_random_envelopes_min_slack": star_min}

# ---------------------------------------------------------------- 7 observation
grid = [Fr(k, 10) for k in range(11)]
tested = bad = 0
while tested < 500:
    vals7 = [grid[int(rng.integers(11))] for _ in range(7)]
    i = make_profile(vals7)
    if not all(three_conditions(i)) or min(star_values(i).values()) < 0:
        continue
    tested += 1
    bad += not attainable(i)
out["F2_observation"] = {"grid_profiles_satisfying_conditions_and_star": tested,
                         "LP_unattainable": bad}
assert bad > 0

# ---------------------------------------------------------------- 8 Proposition 4 example
# I({1}) = I({3}) = 1 forces delta_1, delta_3 into M, hence I(A) = 1 whenever A separates 1 and 3.
ex = make_profile([Fr(4, 5), Fr(1), Fr(9, 10), Fr(1), Fr(19, 20), Fr(17, 20), Fr(3, 10)])
sym, sub, bound = three_conditions(ex)
st = star_values(ex)
assert sym and sub and bound and min(st.values()) > 0
assert ex[frozenset([0, 3])] == Fr(3, 10) and 3 in {0, 3} and 1 not in {0, 3}
assert not attainable(ex)
# Proposition 3 itself on random credal sets containing two Dirac measures
viol = 0
for t in range(1000):
    n_ = int(rng.integers(3, 7))
    P = list(rng.dirichlet(np.ones(n_), size=int(rng.integers(0, 4))))
    x, y = rng.choice(n_, size=2, replace=False)
    P += [np.eye(n_)[x], np.eye(n_)[y]]
    I = envelope(np.array(P), n_)
    for A in allevents(n_):
        if x in A and y not in A and abs(I[A] - 1) > 1e-12:
            viol += 1
assert viol == 0
out["Proposition4_example"] = {"profile": ["4/5", "1", "9/10", "1", "19/20", "17/20", "3/10"],
                               "symmetry": sym, "subadditivity": sub, "bound": bound,
                               "star_slack_by_a": {str(a): str(s) for a, s in st.items()},
                               "LP_attainable": False, "random_check_violations": viol}

# ---------------------------------------------------------------- 9 exact certificates for C3 and C4
# The stored floating-point values of C3 and C4 (results/t1_reduction_hierarchy.json) are exact binary
# rationals; read them as Fractions and certify the claims in exact arithmetic.
T1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "inputs",
                       "t1_reduction_hierarchy.json")
t1 = json.load(open(T1_PATH, encoding="utf-8"))


def parse(d, n_):
    L = {frozenset(json.loads(k)): Fr(v) for k, v in d.items()}
    L[frozenset()] = Fr(0)
    L[frozenset(range(n_))] = Fr(1)
    return L


def two_monotone_violations(L):
    ev = list(L)
    return [(A, B) for A in ev for B in ev if L[A | B] + L[A & B] < L[A] + L[B]]


def coherent_exact(L, n_):
    """Every bound L(A) is attained by a vertex of M(L), found by exact vertex enumeration."""
    full_ = frozenset(range(n_))
    cons = [("ev", B) for B in L if B and B != full_] + [("atom", w) for w in range(n_)]

    def row(c):
        r = [Fr(0)] * n_
        if c[0] == "ev":
            for w in c[1]:
                r[w] = Fr(1)
            return r, L[c[1]]
        r[c[1]] = Fr(1)
        return r, Fr(0)

    def feasible(P):
        return all(p >= 0 for p in P) and sum(P) == 1 and all(sum(P[w] for w in B) >= L[B] for B in L)

    for A in L:
        if not A or A == full_:
            continue
        found = False
        base = [([Fr(1)] * n_, Fr(1)), row(("ev", A))]
        for extra in itertools.combinations(cons, n_ - 2):
            M_ = sp.Matrix([r for r, _ in base] + [row(c)[0] for c in extra])
            if M_.det() == 0:
                continue
            b_ = sp.Matrix([b for _, b in base] + [row(c)[1] for c in extra])
            P = [Fr(int(sp.fraction(x)[0]), int(sp.fraction(x)[1])) for x in M_.LUsolve(b_)]
            if feasible(P) and sum(P[w] for w in A) == L[A]:
                found = True
                break
        if not found:
            return False
    return True


def exact_envelope_from(Lf, n_):
    """For each event A, an LP gives a measure attaining Lf(A); round it to a rational probability with
    denominator 10**9 and return the exact lower envelope of these measures (coherent by construction)."""
    full_ = frozenset(range(n_))
    evs = [A for A in Lf if A and A != full_]
    Ps = []
    for A in evs:
        A_ub = [[-1.0 if w in B else 0.0 for w in range(n_)] for B in evs]
        b_ub = [-float(Lf[B]) for B in evs]
        res = linprog([1.0 if w in A else 0.0 for w in range(n_)], A_ub=A_ub, b_ub=b_ub,
                      A_eq=[[1.0] * n_], b_eq=[1.0], bounds=[(0, None)] * n_, method="highs")
        assert res.status == 0
        q = [Fr(round(x * 10**9), 10**9) for x in res.x]
        q[-1] = 1 - sum(q[:-1])
        assert all(x >= 0 for x in q)
        Ps.append(q)
    L = {A: min(sum(P[w] for w in A) for P in Ps) for A in allevents(n_)}
    return L, Ps


C3f = parse(t1["C3_coherent_not_2monotone_n4"], 4)
C3, C3_P = exact_envelope_from(C3f, 4)
C3_viol = two_monotone_violations(C3)
C3_dist = max(abs(C3[A] - C3f[A]) for A in C3)
assert C3_viol and C3_dist < Fr(1, 10**6)
C4 = parse(t1["C4_2monotone_not_belief_n3"]["L"], 3)
C4_viol = two_monotone_violations(C4)
mob = {}
for A in C4:
    mob[A] = sum((-1) ** (len(A) - len(B)) * C4[frozenset(B)]
                 for k in range(len(A) + 1) for B in itertools.combinations(sorted(A), k))
assert not C4_viol and min(mob.values()) < 0
out["C3_C4_exact"] = {
    "C3_exact_envelope_of": f"{len(C3_P)} rational measures (denominator 10^9), coherent by construction",
    "C3_max_distance_to_stored_values": float(C3_dist),
    "C3_two_monotone_violations_exact": len(C3_viol),
    "C4_two_monotone_exact": True, "C4_mobius_min_exact": float(min(mob.values()))}

with open(os.path.join(RESULTS, "v10_open_problems.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
print("V10 CHECKS PASSED")
