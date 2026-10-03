"""Independent review of Florentin's v9 (Sections 10-12, A.7). ML, 1-oct-2026.
Own implementation (different seed, more copulas) of Theorems 10, 11, Prop. 2, Cor. 3-5
and of the numbers of Sections 11-12."""
import json, itertools, numpy as np, sympy as sp
from scipy.optimize import linprog

rng = np.random.default_rng(777); out = {}
Pi = lambda u, v: u * v
M = lambda u, v: min(u, v)
W = lambda u, v: max(0.0, u + v - 1)
def fgm(t): return lambda u, v: u * v * (1 + t * (1 - u) * (1 - v))
def frank(t): return lambda u, v: -np.log1p(np.expm1(-t * u) * np.expm1(-t * v) / np.expm1(-t)) / t
def clay(t): return lambda u, v: 0.0 if u == 0 or v == 0 else max(u ** -t + v ** -t - 1, 0) ** (-1 / t)
def gum(t): return lambda u, v: 0.0 if u == 0 or v == 0 else float(np.exp(-((-np.log(u)) ** t + (-np.log(v)) ** t) ** (1 / t)))
def plack(t):
    def C(u, v):
        s = 1 + (t - 1) * (u + v)
        return (s - np.sqrt(s * s - 4 * u * v * t * (t - 1))) / (2 * (t - 1))
    return C
def joe(t): return lambda u, v: 1 - ((1 - u) ** t + (1 - v) ** t - ((1 - u) * (1 - v)) ** t) ** (1 / t)
surv = lambda C: (lambda u, v: u + v - 1 + C(1 - u, 1 - v))
COP = {'Pi': (Pi, 1), 'M': (M, 1), 'W': (W, 1), 'FGM(1)': (fgm(1), 1), 'FGM(-0.5)': (fgm(-.5), 1),
       'Frank(1)': (frank(1.), 1), 'Frank(-8)': (frank(-8.), 1), 'Plackett(3)': (plack(3.), 1),
       'Plackett(0.3)': (plack(.3), 1), 'Clayton(0.5)': (clay(.5), 0), 'Clayton(4)': (clay(4.), 0),
       'survGumbel(2)': (surv(gum(2.)), 0), 'Gumbel(3)': (gum(3.), 0), 'Joe(2)': (joe(2.), 0)}
def _clip(C): return lambda u, v: C(min(max(float(u), 0.0), 1.0), min(max(float(v), 0.0), 1.0))
COP = {k: (_clip(C), rs) for k, (C, rs) in COP.items()}
PAT = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
AT = list(itertools.product(range(4), range(4)))

def lpmin(x, y, obj):
    A, b = [], []
    for k in range(3):
        A.append([-PAT[i][k] for i, j in AT]); b.append(-x[k])
        A.append([-PAT[j][k] for i, j in AT]); b.append(-y[k])
    return linprog(obj, A_ub=A, b_ub=b, A_eq=[np.ones(16)], b_eq=[1], bounds=(0, 1), method='highs').fun

def sw_obj(X, c, disj=False):
    o = []
    for i, j in AT:
        a, b = PAT[i][X], PAT[j][X]; I, U = a * b, max(a, b)
        o.append((1 - c) * U + c * I if disj else (1 - c) * I + c * U)
    return np.array(o, float)

def Npl(C, c, x, y):
    T = c * (x[0] + y[0]) + (1 - 2 * c) * C(x[0], y[0])
    F = (1 - c) * (x[2] + y[2]) - (1 - 2 * c) * C(x[2], y[2])
    return np.array([T, (x[1] + y[1]) / 2, F])

# ---- Theorem 10(b): LP on 16 atoms, no dependence assumption
e = 0
for _ in range(3000):
    x, y = rng.random(3), rng.random(3)
    c = rng.choice([0, .5, 1]) if rng.random() < .15 else rng.random()
    lp = np.array([lpmin(x, y, sw_obj(0, c)), lpmin(x, y, sw_obj(1, .5)), lpmin(x, y, sw_obj(2, c, True))])
    if c <= .5: f = np.array([Npl(W, c, x, y)[0], (x[1] + y[1]) / 2, Npl(M, c, x, y)[2]])
    else:       f = np.array([Npl(M, c, x, y)[0], (x[1] + y[1]) / 2, Npl(W, c, x, y)[2]])
    e = max(e, abs(lp - f).max())
out['T10b_LP_3000_maxerr'] = e

# ---- Theorem 10(a): lower bound by monotonicity along r=C(p,q); explicit attaining Q in M_C
def Qattain(x, y, X, r):
    a, b = x[X], y[X]; Q = np.zeros((4, 4))
    Q[3, 3] = r; Q[3, X] = a - r; Q[X, 3] = b - r; Q[X, X] = 1 - a - b + r
    return Q
mono = 0; att = 0
for name, (C, _) in COP.items():
    for _ in range(400):
        x, y, c = rng.random(3), rng.random(3), rng.random()
        for X, cc, dj in [(0, c, 0), (1, .5, 0), (2, c, 1)]:
            def g(p, q):
                I = C(p, q); U = p + q - I
                return (1 - cc) * U + cc * I if dj else (1 - cc) * I + cc * U
            base = g(x[X], y[X])
            for _ in range(30):
                p = x[X] + (1 - x[X]) * rng.random(); q = y[X] + (1 - y[X]) * rng.random()
                if g(p, q) < base - 1e-10: mono += 1
            Q = Qattain(x, y, X, C(x[X], y[X]))
            assert Q.min() > -1e-12
            for Y in range(3):
                pA = sum(Q[i, j] for i, j in AT if PAT[i][Y]); pB = sum(Q[i, j] for i, j in AT if PAT[j][Y])
                r = sum(Q[i, j] for i, j in AT if PAT[i][Y] and PAT[j][Y])
                assert pA >= x[Y] - 1e-12 and pB >= y[Y] - 1e-12 and abs(r - C(pA, pB)) < 1e-9, (name, X, Y, x, y, pA, pB, r, C(pA, pB))
            val = (sw_obj(X, cc, bool(dj)) * Q.flatten()).sum()
            att = max(att, abs(val - Npl(C, c, x, y)[X]))
out['T10a_copulas'] = list(COP); out['T10a_pairs_per_copula'] = 400
out['T10a_monotonicity_violations'] = mono; out['T10a_attainment_maxerr'] = att
# sign of the truth difference M vs W for c>1/2 (A.7(c) says "positive")
x, y = np.array([.5, .2, .3]), np.array([.6, .1, .3])
out['T10c_truth_M_minus_W_at_c=0.8'] = round(Npl(M, .8, x, y)[0] - Npl(W, .8, x, y)[0], 4)
out['T10c_truth_M_minus_W_at_c=0.2'] = round(Npl(M, .2, x, y)[0] - Npl(W, .2, x, y)[0], 4)

# ---- Theorem 11 on the classical frame: brute-force bounds of the switched conjunction
def simplex():
    v = rng.exponential(size=3); return v / v.sum()
t11 = {}
for name, (C, rs) in COP.items():
    Ch = surv(C); eb = ef = dF = es = 0
    for _ in range(1500):
        x, y, c = simplex(), simplex(), rng.random(); u1, u2 = 1 - x[2], 1 - y[2]
        ps = np.linspace(x[0], u1, 25); qs = np.linspace(y[0], u2, 25)
        vals = [c * (p + q) + (1 - 2 * c) * C(p, q) for p in ps for q in qs]; L, U = min(vals), max(vals)
        N = Npl(C, c, x, y); eb = max(eb, abs(N[0] - L))
        d = N[2] - (1 - U); dF = max(dF, abs(d))
        ef = max(ef, abs(d - (1 - 2 * c) * (Ch(x[2], y[2]) - C(x[2], y[2]))))
        es = max(es, abs(N[1] - (U - L) - (1 - 2 * c) * ((x[1] + y[1]) / 2 - (C(u1, u2) - C(x[0], y[0])))))
    t11[name] = dict(radially_symmetric=bool(rs), truth_err=eb, falsity_formula_err=ef,
                     max_falsity_diff=dF, sigma_formula_err=es)
out['T11_bruteforce_1500_per_copula'] = t11

# ---- symbolic checks
c, a, b, d, F1, F2, Cf, Cs = sp.symbols('c a b d F1 F2 Cf Cs')
f = lambda u, v: c * (u + v) + (1 - 2 * c) * u * v
out['P2a_symbolic_residual'] = str(sp.simplify(sp.expand(f(f(a, b), d) - f(a, f(b, d)) - c * (1 - c) * (d - a))))
U = c * ((1 - F1) + (1 - F2)) + (1 - 2 * c) * Cs; Fpl = (1 - c) * (F1 + F2) - (1 - 2 * c) * Cf
out['T11b_symbolic_residual'] = str(sp.simplify(sp.expand(Fpl - (1 - U) - (1 - 2 * c) * (F1 + F2 - 1 + Cs - Cf))))

# ---- Prop 2: n-ary switched conjunction, no assumption: LP vs min of the two closed-form candidates
def lp_nary(p, c):
    n = len(p); P = list(itertools.product([0, 1], repeat=n))
    return linprog([(1 - c) * all(s) + c * any(s) for s in P], A_eq=[[s[i] for s in P] for i in range(n)] + [[1] * len(P)],
                   b_eq=list(p) + [1], bounds=(0, 1), method='highs').fun
gap = []
for _ in range(400):
    n = rng.integers(3, 6); p = rng.random(n); c = rng.random()
    cand1 = (1 - c) * max(0, p.sum() - (n - 1)) + c * min(1, p.sum()); cand2 = (1 - c) * p.min() + c * p.max()
    gap.append(min(cand1, cand2) - lp_nary(p, c))
gap = np.array(gap)
out['P2_nary_min_candidates_minus_LP'] = dict(max=float(gap.max()), min=float(gap.min()), share_positive=float(np.mean(gap > 1e-9)))

# ---- minimal lifting vs product lifting for pooled (mixture) events: Sections 11.3, 11.4
def minlift_mixture(xs, w):
    m = len(xs); A = np.array([1 - np.eye(m, dtype=int)[k] for k in range(m)] + [np.ones(m, int)]).T
    obj = np.zeros(m + 1)
    for k, wk in w.items(): obj += wk * A[k]
    return linprog(obj, A_ub=-A, b_ub=-np.array(xs), A_eq=[np.ones(m + 1)], b_eq=[1], bounds=(0, 1), method='highs').fun
fc = [0.62, 0.30, 0.08, 0.55, 0.20, 0.35, 0.70, 0.10, 0.60]
out['A3_equal_pool_T'] = dict(weighted_sum=round(np.mean([.62, .55, .70]), 4),
                              minimal_lifting_10_atoms=round(minlift_mixture(fc, {0: 1/3, 3: 1/3, 6: 1/3}), 4))
out['A3_contradiction_pool_T'] = dict(weighted_sum=round(.5 * .62 + .55 / 3 + .7 / 6, 4),
                                      minimal_lifting_10_atoms=round(minlift_mixture(fc, {0: .5, 3: 1/3, 6: 1/6}), 4))
neu = [0.5, 0.1, 0.2, 0.6, 0.2, 0.4, 0.8, 0.0, 0.1, 0.4, 0.3, 0.5]
out['A4_graduation_I_mean'] = dict(mean_of_lowers=0.15,
                                   minimal_lifting_13_atoms=round(minlift_mixture(neu, {1: .25, 4: .25, 7: .25, 10: .25}), 4))
# conjunction of the four I events on the minimal lifting (forced Lukasiewicz-type structure)
m = 12; A = np.array([1 - np.eye(m, dtype=int)[k] for k in range(m)] + [np.ones(m, int)]).T

# ---- Cor 3(b): minimality is for the full cube; a fuzzy value + an IF pair (T+F<=1) needs fewer atoms
def faithful(pats, pts, m):
    A = np.array(pats).T
    for x in pts:
        for k in range(m):
            r = linprog(A[k], A_ub=-A, b_ub=-x, A_eq=[np.ones(len(pats))], b_eq=[1], bounds=(0, 1), method='highs')
            if r.status != 0 or abs(r.fun - x[k]) > 1e-9: return False
    return True
pts = []
while len(pts) < 150:
    v = rng.random(3)
    if v[1] + v[2] <= 1: pts.append(v)
pts += [np.array(v, float) for v in [(1, 1, 0), (1, 0, 1), (0, 1, 0), (0, 0, 1), (1, 0, 0), (0, 0, 0), (1, .5, .5)]]
allp = [p for p in itertools.product([0, 1], repeat=3) if any(p)]
out['C3b_fuzzy_plus_IFpair_faithful_3atom_sets'] = [list(map(list, S)) for S in itertools.combinations(allp, 3) if faithful(S, pts, 3)]

# ---- Section 12: segment K: its lower envelope is a belief function
P1 = np.array([.2, .3, .5]); P2 = np.array([.4, .5, .1])
ev = [s for r in (1, 2, 3) for s in itertools.combinations(range(3), r)]
L = {s: min(P1[list(s)].sum(), P2[list(s)].sum()) for s in ev}
mob = {s: sum((-1) ** (len(s) - len(t)) * L[t] for r in range(1, len(s) + 1) for t in itertools.combinations(s, r)) for s in ev}
out['Sec12_segment_lower_envelope_mobius'] = {str(k): round(v, 4) for k, v in mob.items()}

# ---- Zadeh's example (Table 11)
m1 = {'A': .9, 'C': .1}; m2 = {'B': .9, 'C': .1}
conf = sum(m1[a] * m2[b] for a in m1 for b in m2 if a != b)
pcr = {'A': 0, 'B': 0, 'C': m1['C'] * m2['C']}
for a_ in m1:
    for b_ in m2:
        if a_ != b_:
            k = m1[a_] * m2[b_]; pcr[a_] += m1[a_] * k / (m1[a_] + m2[b_]); pcr[b_] += m2[b_] * k / (m1[a_] + m2[b_])
ev3 = [s for r in (1, 2) for s in itertools.combinations(range(3), r)]
A_, bb = [], []
for P in [np.array([.9, 0, .1]), np.array([0, .9, .1])]:
    for s in ev3:
        row = [1 if i in s else 0 for i in range(3)]; v = P[list(s)].sum()
        A_.append([-z for z in row] + [-1]); bb.append(-v); A_.append(row + [-1]); bb.append(v)
r = linprog([0, 0, 0, 1], A_ub=A_, b_ub=bb, A_eq=[[1, 1, 1, 0]], b_eq=[1], bounds=[(0, 1)] * 3 + [(0, None)], method='highs')
out['Zadeh'] = dict(conflict=round(conf, 4), dempster_mC=round(m1['C'] * m2['C'] / (1 - conf), 4),
                    PCR5={k: round(v, 4) for k, v in pcr.items()}, delta_LP=round(r.fun, 4))

json.dump(out, open('verify_v9_plithogenic.json', 'w'), indent=1, default=float)  # book edition: cwd
print(json.dumps(out, indent=1, default=float))
