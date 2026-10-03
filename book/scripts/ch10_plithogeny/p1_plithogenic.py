"""p1_plithogenic.py -- checks for Section 10 (plithogenic probability and credal sets).

Theorem 10: plithogenic N-norms as natural extensions on the product of two minimal liftings.
Theorem 11: plithogenic N-norms on the classical frame (radial symmetry, surplus).
Corollary 3: plithogenic lifting (1 + sum m_k atoms).
Corollary 4: plithogenic IDM with indeterminate observations.
Section 11: numbers of the worked applications.
"""
import json, itertools
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(20260930)
out = {}

# ---------------------------------------------------------------- copulas
def Pi(u, v): return u * v
def M(u, v): return min(u, v)
def W(u, v): return max(0.0, u + v - 1)
def fgm(t): return lambda u, v: u * v * (1 + t * (1 - u) * (1 - v))
def frank(t):
    def C(u, v):
        return -np.log1p((np.expm1(-t * u) * np.expm1(-t * v)) / np.expm1(-t)) / t
    return C
def clayton(t):
    def C(u, v):
        if u == 0 or v == 0: return 0.0
        return max(u ** -t + v ** -t - 1, 0) ** (-1 / t)
    return C
def gumbel(t):
    def C(u, v):
        if u == 0 or v == 0: return 0.0
        return np.exp(-((-np.log(u)) ** t + (-np.log(v)) ** t) ** (1 / t))
    return C
def survival(C): return lambda u, v: u + v - 1 + C(1 - u, 1 - v)

COPULAS = {
    "Pi": (Pi, True), "M": (M, True), "W": (W, True),
    "FGM(0.5)": (fgm(0.5), True), "FGM(-1)": (fgm(-1), True),
    "Frank(5)": (frank(5.0), True), "Frank(-5)": (frank(-5.0), True),
    "Clayton(2)": (clayton(2.0), False), "Clayton(8)": (clayton(8.0), False),
    "survClayton(2)": (survival(clayton(2.0)), False), "Gumbel(2)": (gumbel(2.0), False),
}

def pl_nnorm(C, c, x, y):
    """plithogenic N-norm with copula C and contradiction degree c (on T and F); I averaged."""
    T1, I1, F1 = x; T2, I2, F2 = y
    S = lambda a, b: a + b - C(a, b)
    T = (1 - c) * C(T1, T2) + c * S(T1, T2)
    I = 0.5 * (C(I1, I2) + S(I1, I2))
    F = (1 - c) * S(F1, F2) + c * C(F1, F2)
    return np.array([T, I, F])

# ------------------------------------------------------- minimal lifting (m=3)
PAT = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]   # order (T,I,F)
ATOMS = list(itertools.product(range(4), range(4)))  # 16 atoms of the product

def lp_min_switched(x, y, X, c, mode):
    """min over joints Q (no dependence assumption) whose marginals lie in K(x), K(y)
    of the switched event probability. mode 'conj': (1-c)Q(A&B)+cQ(AvB); 'disj': (1-c)Q(AvB)+cQ(A&B)."""
    obj = np.zeros(16)
    for k, (i, j) in enumerate(ATOMS):
        a, b = PAT[i][X], PAT[j][X]
        inter, union = a * b, max(a, b)
        obj[k] = (1 - c) * inter + c * union if mode == "conj" else (1 - c) * union + c * inter
    A_ub, b_ub = [], []
    for comp in range(3):  # marginal lower bounds: -Q(E) <= -x
        A_ub.append([-PAT[i][comp] for (i, j) in ATOMS]); b_ub.append(-x[comp])
        A_ub.append([-PAT[j][comp] for (i, j) in ATOMS]); b_ub.append(-y[comp])
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=[np.ones(16)], b_eq=[1], bounds=(0, 1), method="highs")
    return res.fun

def nodep_formula(c, x, y):
    T1, I1, F1 = x; T2, I2, F2 = y
    if c <= 0.5:
        T = (1 - c) * W(T1, T2) + c * min(1, T1 + T2)
        F = (1 - c) * max(F1, F2) + c * min(F1, F2)
    else:
        T = (1 - c) * min(T1, T2) + c * max(T1, T2)
        F = (1 - c) * min(1, F1 + F2) + c * W(F1, F2)
    return np.array([T, (I1 + I2) / 2, F])

# Theorem 10(b): no dependence assumption, LP on 16 atoms
err_b = 0.0
for _ in range(1500):
    x, y = rng.random(3), rng.random(3)          # arbitrary triples of the cube
    c = rng.random() if rng.random() < 0.9 else rng.choice([0, 0.5, 1])
    lp = np.array([lp_min_switched(x, y, 0, c, "conj"),
                   lp_min_switched(x, y, 1, 0.5, "conj"),
                   lp_min_switched(x, y, 2, c, "disj")])
    err_b = max(err_b, np.abs(lp - nodep_formula(c, x, y)).max())
out["T10b_nodep_LP_pairs"] = 1500
out["T10b_max_error"] = err_b
assert err_b < 1e-8, err_b

# Theorem 10(a): event-wise C-coupling; lower envelope attained at lower endpoints
def switched(C, c, p, q, mode):
    S = p + q - C(p, q)
    return (1 - c) * C(p, q) + c * S if mode == "conj" else (1 - c) * S + c * C(p, q)
err_a, mono_viol = 0.0, 0
grid = np.linspace(0, 1, 21)
for name, (C, _) in COPULAS.items():
    for _ in range(300):
        x, y, c = rng.random(3), rng.random(3), rng.random()
        target = pl_nnorm(C, c, x, y)
        vals = []
        for X, cc, mode in [(0, c, "conj"), (1, 0.5, "conj"), (2, c, "disj")]:
            ps = x[X] + (1 - x[X]) * grid; qs = y[X] + (1 - y[X]) * grid
            v = min(switched(C, cc, p, q, mode) for p in ps for q in qs)
            vals.append(v)
            if v < switched(C, cc, x[X], y[X], mode) - 1e-9: mono_viol += 1
        err_a = max(err_a, np.abs(np.array(vals) - target).max())
out["T10a_copulas"] = list(COPULAS)
out["T10a_max_error"] = err_a
out["T10a_monotonicity_violations"] = mono_viol
assert err_a < 1e-9 and mono_viol == 0

# Theorem 10(c): at c = 1/2 every copula gives the componentwise average
err_c = 0.0
for name, (C, _) in COPULAS.items():
    for _ in range(500):
        x, y = rng.random(3), rng.random(3)
        err_c = max(err_c, np.abs(pl_nnorm(C, 0.5, x, y) - (x + y) / 2).max())
out["T10c_max_error"] = err_c
assert err_c < 1e-12

# ------------------------------------------------ Theorem 11 (classical frame)
def simplex():
    e = rng.exponential(size=3); return e / e.sum()
res11 = {}
for name, (C, rs) in COPULAS.items():
    Chat = survival(C)
    maxF_err_formula, maxF_diff, max_sig_err, sig_min, sig_max, T_err = 0, 0, 0, 1, -1, 0
    for _ in range(4000):
        x, y, c = simplex(), simplex(), rng.random()
        T1, I1, F1 = x; T2, I2, F2 = y
        u1, u2 = 1 - F1, 1 - F2
        L = c * (T1 + T2) + (1 - 2 * c) * C(T1, T2)            # credal lower of switched conj.
        U = c * (u1 + u2) + (1 - 2 * c) * C(u1, u2)            # credal upper
        pl = pl_nnorm(C, c, x, y)
        T_err = max(T_err, abs(pl[0] - L))
        dF = pl[2] - (1 - U)
        maxF_err_formula = max(maxF_err_formula, abs(dF - (1 - 2 * c) * (Chat(F1, F2) - C(F1, F2))))
        maxF_diff = max(maxF_diff, abs(dF))
        sig = pl[1] - (U - L)
        max_sig_err = max(max_sig_err, abs(sig - (1 - 2 * c) * ((I1 + I2) / 2 - (C(u1, u2) - C(T1, T2)))))
        sig_min, sig_max = min(sig_min, sig), max(sig_max, sig)
    res11[name] = dict(radially_symmetric=rs, T_err=T_err, F_formula_err=maxF_err_formula,
                       max_abs_F_diff=maxF_diff, sigma_formula_err=max_sig_err,
                       sigma_range=[sig_min, sig_max])
    assert T_err < 1e-12 and maxF_err_formula < 1e-9 and max_sig_err < 1e-9
    if rs: assert maxF_diff < 1e-9, (name, maxF_diff)
    else: assert maxF_diff > 1e-4, (name, maxF_diff)
out["T11"] = res11

# --------------------------------------------- Corollary 3 (plithogenic lifting)
def faithful_check(ms, trials=200):
    m = sum(ms)
    pats = [tuple(1 - np.eye(m, dtype=int)[k]) for k in range(m)] + [tuple([1] * m)]
    A = np.array(pats).T  # m x (m+1)
    worst = 0.0
    for _ in range(trials):
        x = rng.random(m)
        for k in range(m):
            r = linprog(A[k], A_ub=-A, b_ub=-x, A_eq=[np.ones(m + 1)], b_eq=[1], bounds=(0, 1), method="highs")
            worst = max(worst, abs(r.fun - x[k]))
    return worst
out["C3_attributes_m"] = [3, 3, 3]
out["C3_atoms"] = 1 + 9
out["C3_max_error"] = faithful_check([3, 3, 3])
out["C3_hybrid_m"] = [1, 2, 3]   # classical, intuitionistic-fuzzy, neutrosophic attribute
out["C3_hybrid_max_error"] = faithful_check([1, 2, 3])
assert out["C3_max_error"] < 1e-9 and out["C3_hybrid_max_error"] < 1e-9

# --------------------------------------------- Corollary 4 (plithogenic IDM)
def singleton_checks(T, I):
    D = 1 - T.sum()
    proper = (D >= -1e-12) and (D <= I.sum() + 1e-12)
    reach = all(I[i] <= D + 1e-12 and D <= I.sum() - I[i] + 1e-12 for i in range(len(T)))
    return proper, reach
def lp_bounds(l, u, A):
    k = len(l); cvec = np.array([1.0 if i in A else 0.0 for i in range(k)])
    lo = linprog(cvec, A_eq=[np.ones(k)], b_eq=[1], bounds=list(zip(l, u)), method="highs").fun
    hi = -linprog(-cvec, A_eq=[np.ones(k)], b_eq=[1], bounds=list(zip(l, u)), method="highs").fun
    return lo, hi
err4, nonco = 0.0, 0
for _ in range(2000):
    k = rng.integers(2, 6); n = rng.integers(0, 30, size=k); nq = rng.integers(0, 20); s = rng.uniform(0.5, 4)
    N = n.sum() + nq
    T = n / (N + s); I = np.full(k, (nq + s) / (N + s))
    p, r = singleton_checks(T, I)
    if not (p and r): nonco += 1
    A = set(rng.choice(k, size=rng.integers(1, k), replace=False).tolist())
    lo, hi = lp_bounds(T, T + I, A)
    nA = sum(n[i] for i in A)
    err4 = max(err4, abs(lo - nA / (N + s)), abs(hi - (nA + nq + s) / (N + s)))
out["C4_cases"] = 2000; out["C4_noncoherent"] = nonco; out["C4_max_error"] = err4
assert nonco == 0 and err4 < 1e-9

# ============================================================ applications
app = {}
# A1: diagnosis -- conjunction of two criteria, varying contradiction degree
x = np.array([0.70, 0.20, 0.10])   # biomarker panel (reference laboratory)
y = np.array([0.55, 0.25, 0.20])   # imaging read by a second centre
rows = []
for c in [0, 0.25, 0.5, 0.75, 1]:
    ind = pl_nnorm(Pi, c, x, y)
    nod = nodep_formula(c, x, y)
    como = pl_nnorm(M, c, x, y)
    rows.append(dict(c=c, independence=ind.round(4).tolist(), comonotone=como.round(4).tolist(),
                     no_assumption=nod.round(4).tolist(),
                     T_gap=round(float(max(ind[0], como[0], nod[0]) - min(ind[0], como[0], nod[0])), 4)))
app["A1_inputs"] = dict(x=x.tolist(), y=y.tolist()); app["A1_rows"] = rows

# A2: plithogenic IDM -- satisfaction survey, attribute 'region'
regions = {"North": (112, 38, 20), "Centre": (95, 30, 45), "South": (60, 52, 8)}   # (sat, not sat, indeterminate)
s = 2.0; a2 = {}
for rg, (ns, nn, nq) in regions.items():
    N = ns + nn + nq
    T = ns / (N + s); F = nn / (N + s); I = 1 - T - F
    a2[rg] = dict(N=N, T=round(T, 4), I=round(I, 4), F=round(F, 4), interval=[round(T, 4), round(1 - F, 4)],
                  interval_if_indeterminate_dropped=[round(ns / (ns + nn + s), 4), round((ns + s) / (ns + nn + s), 4)])
app["A2"] = a2

# A3: fact-checking -- three source types, contradiction degrees to the dominant (official) type
src = {"official": ((0.62, 0.30, 0.08), 0.0), "press": ((0.55, 0.20, 0.35), 1 / 3), "social": ((0.70, 0.10, 0.60), 2 / 3)}
two_atom = {k: [v[0][0], 1 - v[0][2]] for k, v in src.items()}
# glut mass needed on the minimal lifting of each type: T+F-1 when positive
glut = {k: round(max(0.0, v[0][0] + v[0][2] - 1), 4) for k, v in src.items()}
# pooled assessments: equal weights (c = 1/2 reading) and contradiction weights w_k proportional to 1 - c_k
V = np.array([v[0] for v in src.values()]); cs = np.array([v[1] for v in src.values()])
eq = V.mean(axis=0); w = (1 - cs) / (1 - cs).sum(); cw = w @ V
app["A3"] = dict(two_atom_intervals=two_atom, glut_mass=glut,
                 delta_social=round((0.70 + 0.60 - 1) / 2, 4),
                 equal_pool=eq.round(4).tolist(), equal_pool_interval=[round(eq[0], 4), round(1 - eq[2], 4)],
                 weights=w.round(4).tolist(), contradiction_pool=cw.round(4).tolist(),
                 contradiction_pool_interval=[round(cw[0], 4), round(1 - cw[2], 4)])
out["applications"] = app


# ------------------------------------------ Proposition 2 (many attribute values)
def f_pi(c, a, b): return c * (a + b) + (1 - 2 * c) * a * b
err = 0.0
for _ in range(10000):
    a, b, dd, c = rng.random(4)
    err = max(err, abs(f_pi(c, f_pi(c, a, b), dd) - f_pi(c, a, f_pi(c, b, dd)) - c * (1 - c) * (dd - a)))
out["P2a_assoc_defect_formula_err"] = err; assert err < 1e-12
nonassoc = {}
for name, C in [("M", M), ("W", W)]:
    g = lambda c, a, b: c * (a + b) + (1 - 2 * c) * C(a, b)
    nonassoc[name] = max(abs(g(c, g(c, a, b), dd) - g(c, a, g(c, b, dd)))
                         for a, b, dd, c in rng.random((20000, 4)))
out["P2a_max_assoc_defect_M_W"] = nonassoc
# (b) n-ary switched conjunction: closed forms are lower envelopes (monotone) -- sampled check
viol = 0
for _ in range(3000):
    n = rng.integers(3, 6); x = rng.random(n); c = rng.random()
    ind = (1 - c) * x.prod() + c * (1 - (1 - x).prod())
    com = (1 - c) * x.min() + c * x.max()
    for _ in range(20):
        p = x + (1 - x) * rng.random(n)
        if (1 - c) * p.prod() + c * (1 - (1 - p).prod()) < ind - 1e-12: viol += 1
        if (1 - c) * p.min() + c * p.max() < com - 1e-12: viol += 1
out["P2b_monotonicity_violations"] = viol; assert viol == 0
# (c) n >= 3: the 1/2-switch is not dependence-free; the mean is
x = np.array([0.3, 0.6, 0.8])
out["P2c_example"] = dict(x=x.tolist(), independence=round(0.5 * (x.prod() + 1 - (1 - x).prod()), 4),
                          comonotone=round(0.5 * (x.min() + x.max()), 4), mean=round(x.mean(), 4))
def lp_nary(p, c):
    n = len(p); pats = list(itertools.product([0, 1], repeat=n))
    obj = [(1 - c) * all(s) + c * any(s) for s in pats]
    Aeq = [[s[i] for s in pats] for i in range(n)] + [[1] * len(pats)]
    return linprog(obj, A_eq=Aeq, b_eq=list(p) + [1], bounds=(0, 1), method="highs").fun
def lp_sel(p):  # random-selection event: mean of marginals, any coupling
    n = len(p); pats = list(itertools.product([0, 1], repeat=n))
    obj = [np.mean(s) for s in pats]
    Aeq = [[s[i] for s in pats] for i in range(n)] + [[1] * len(pats)]
    lo = linprog(obj, A_eq=Aeq, b_eq=list(p) + [1], bounds=(0, 1), method="highs").fun
    hi = -linprog(-np.array(obj), A_eq=Aeq, b_eq=list(p) + [1], bounds=(0, 1), method="highs").fun
    return lo, hi
spread_half, sel_err = 0.0, 0.0
for _ in range(500):
    n = rng.integers(3, 6); p = rng.random(n)
    lo = lp_nary(p, 0.5); hi = -linprog(-np.array([0.5 * all(s) + 0.5 * any(s) for s in itertools.product([0, 1], repeat=n)]),
        A_eq=[[s[i] for s in itertools.product([0, 1], repeat=n)] for i in range(n)] + [[1] * 2 ** n],
        b_eq=list(p) + [1], bounds=(0, 1), method="highs").fun
    spread_half = max(spread_half, hi - lo)
    a, b = lp_sel(p); sel_err = max(sel_err, abs(a - p.mean()), abs(b - p.mean()))
out["P2c_max_spread_half_switch_n>=3"] = spread_half
out["P2c_selection_mean_err"] = sel_err
assert spread_half > 0.05 and sel_err < 1e-9

# ============================================ applications A4, A5
# A4 Jenifer (Smarandache 2017, pp. 135-137): graduation = pass all four courses
def fr_lo(v): return max(0.0, v.sum() - (len(v) - 1))
p = np.array([0.5, 0.6, 0.8, 0.4])
lp_lo = lp_nary(p, 0.0)
lp_hi = -linprog(-np.array([all(s) for s in itertools.product([0, 1], repeat=4)], dtype=float),
    A_eq=[[s[i] for s in itertools.product([0, 1], repeat=4)] for i in range(4)] + [[1] * 16],
    b_eq=list(p) + [1], bounds=(0, 1), method="highs").fun
assert abs(lp_lo - fr_lo(p)) < 1e-9 and abs(lp_hi - p.min()) < 1e-9
def models(l, u):
    r = lambda x: round(float(x), 4)
    return dict(independence=[r(l.prod()), r(u.prod())],
                structured=[r(min(l[:2]) * min(l[2:])), r(min(u[:2]) * min(u[2:]))],
                comonotone=[r(l.min()), r(u.min())],
                no_assumption=[r(fr_lo(l)), r(u.min())])
fuzzy = models(p, p)
ivf = np.array([[0.4, 0.6], [0.3, 0.7], [0.8, 0.9], [0.2, 0.5]])
ifz = np.array([[0.5, 0.2], [0.6, 0.4], [0.8, 0.1], [0.4, 0.5]])
neu = np.array([[0.5, 0.1, 0.2], [0.6, 0.2, 0.4], [0.8, 0.0, 0.1], [0.4, 0.3, 0.5]])
iv = models(ivf[:, 0], ivf[:, 1])
if_ = models(ifz[:, 0], 1 - ifz[:, 1])
ne = models(neu[:, 0], 1 - neu[:, 2])
assert if_ == ne     # Theorem 4: the classical frame cannot tell the intuitionistic and neutrosophic versions apart
I = neu[:, 1]
glut_I = dict(mean=round(float(I.mean()), 4),
              half_switch_independence=round(float(0.5 * (I.prod() + 1 - (1 - I).prod())), 4),
              half_switch_comonotone=round(float(0.5 * (I.min() + I.max())), 4))
# faithfulness of the four neutrosophic triples on the minimal lifting (12 components, 13 atoms)
m = 12; pats = [tuple(1 - np.eye(m, dtype=int)[k]) for k in range(m)] + [tuple([1] * m)]
A = np.array(pats).T; x = neu.flatten(); worst = 0
for k in range(m):
    r_ = linprog(A[k], A_ub=-A, b_ub=-x, A_eq=[np.ones(m + 1)], b_eq=[1], bounds=(0, 1), method="highs")
    worst = max(worst, abs(r_.fun - x[k]))
assert worst < 1e-9
app["A4"] = dict(fuzzy=fuzzy, interval_fuzzy=iv, intuitionistic=if_, neutrosophic_classical=ne,
                 neutrosophic_sums=[round(float(v), 2) for v in neu.sum(axis=1)],
                 lifting_I_of_graduation=glut_I, lifting_faithful_error=worst)

# Corollary 5: the three neutrosophic conjunctions of Smarandache (2019b, eqs. 171, 174, 178)
# = switch vectors (0, g_I, 1) with g_I = 0, 1, 1/2 ; check against LP with no dependence assumption
def lp_switch(x, y, X, g):
    return lp_min_switched(x, y, X, g, "conj")
errs = 0.0; spread = {0: 0.0, 1: 0.0, 0.5: 0.0}
for _ in range(500):
    x, y = rng.random(3), rng.random(3)
    for g in [0, 1, 0.5]:
        lo = lp_switch(x, y, 1, g)
        form = g * (x[1] + y[1]) + (1 - 2 * g) * (W(x[1], y[1]) if g <= 0.5 else M(x[1], y[1]))
        errs = max(errs, abs(lo - form))
        vals = [g * (x[1] + y[1]) + (1 - 2 * g) * C(x[1], y[1]) for C in (W, Pi, M)]
        spread[g] = max(spread[g], max(vals) - min(vals))
out["C5_LP_error"] = errs; out["C5_I_spread_over_W_Pi_M"] = {str(k): v for k, v in spread.items()}
assert errs < 1e-8 and spread[0.5] < 1e-12 and spread[0] > 0.1 and spread[1] > 0.1

# Plithogenic cognitive map step (Martin & Smarandache 2020): per edge the product plithogenic conjunction,
# then the maximum over incoming edges = lower probability of the union with no dependence assumption
def lp_union_lower(ps):
    n = len(ps); pats_ = list(itertools.product([0, 1], repeat=n))
    obj = [float(any(s_)) for s_ in pats_]
    Aeq = [[s_[i] for s_ in pats_] for i in range(n)] + [[1] * len(pats_)]
    return linprog(obj, A_eq=Aeq, b_eq=list(ps) + [1], bounds=(0, 1), method="highs").fun
e_err = 0.0
for _ in range(300):
    n = rng.integers(2, 6); ps = rng.random(n)
    e_err = max(e_err, abs(lp_union_lower(ps) - ps.max()))
out["PCM_max_equals_union_lower_err"] = e_err; assert e_err < 1e-9
# the first step of the published example: node P2 (c = 1/5), source P1 on, edge weight 0.5
out["PCM_example_first_step"] = round(f_pi(0.2, 1.0, 0.5), 4)

# A5 plithogenic fuzzy hypersoft selection: 3 candidates x 4 attribute-value tuples
cand = {"o1": [0.90, 0.90, 0.90, 0.55], "o2": [0.70, 0.70, 0.70, 0.70], "o3": [0.95, 0.95, 0.60, 0.95]}
a5 = {}
for k, v in cand.items():
    v = np.array(v)
    a5[k] = dict(no_assumption_conj=round(fr_lo(v), 4), min_conj=round(v.min(), 4), mean=round(v.mean(), 4),
                 max_disj=round(v.max(), 4), independence_conj=round(v.prod(), 4))
app["A5"] = a5
out["applications"] = app

json.dump(out, open("p1_plithogenic.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
print("ALL CHECKS PASSED")
