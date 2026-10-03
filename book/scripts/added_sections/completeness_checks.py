"""Scripted sanity checks for Section 12.6 "Towards a complete neutrosophic probability logic".

Every check is numbered as in VERIFICACION_COMPLETITUD.md. Nothing here proves a theorem; the checks look for
counterexamples to the stated claims and confirm the worked examples.
Usage: python completeness_checks.py  -> completeness_checks.json and completeness_log.txt
"""
import itertools, json, os, random
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.getcwd()  # book edition: outputs go to the current working directory
rng = random.Random(20261001)
nrng = np.random.default_rng(20261001)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


# ---------------------------------------------------------------- formulas
def rand_formula(n_atoms, depth):
    if depth == 0 or rng.random() < 0.25:
        return ("atom", rng.randrange(n_atoms))
    op = rng.choice(["not", "and", "or"])
    if op == "not":
        return ("not", rand_formula(n_atoms, depth - 1))
    return (op, rand_formula(n_atoms, depth - 1), rand_formula(n_atoms, depth - 1))


def ev_classical(f, w):
    t = f[0]
    if t == "atom": return w[f[1]]
    if t == "true": return True
    if t == "not": return not ev_classical(f[1], w)
    if t == "and": return ev_classical(f[1], w) and ev_classical(f[2], w)
    return ev_classical(f[1], w) or ev_classical(f[2], w)


def ev_lift(f, w):
    """w: tuple of patterns (t,i,f) bits per atom. Returns (T,I,F) bits of the compound formula.
    Conjunction: T = T1 and T2, I = I1 or I2, F = F1 or F2 (cylinder events of Theorem 9);
    negation swaps T and F and keeps I (complement duality (D) = Rivieccio's N2); disjunction by De Morgan."""
    t = f[0]
    if t == "atom": return tuple(bool(b) for b in w[f[1]])
    if t == "true": return (True, False, False)
    if t == "not":
        a = ev_lift(f[1], w); return (a[2], a[1], a[0])
    a, b = ev_lift(f[1], w), ev_lift(f[2], w)
    if t == "and": return (a[0] and b[0], a[1] or b[1], a[2] or b[2])
    return (a[0] or b[0], a[1] or b[1], a[2] and b[2])


def NOT(f): return ("not", f)
def AND(f, g): return ("and", f, g)
def OR(f, g): return ("or", f, g)
P, Q = ("atom", 0), ("atom", 1)

# ---------------------------------------------------------------- credal sets (normalised logic L_NP)
def classical_worlds(n): return list(itertools.product([False, True], repeat=n))


def rand_credal(nw, k):
    return [nrng.dirichlet(np.ones(nw) * rng.choice([0.2, 1.0, 5.0])) for _ in range(k)]


def lower(Pset, mask): return min(float(p[mask].sum()) for p in Pset)
def upper(Pset, mask): return max(float(p[mask].sum()) for p in Pset)


def ext(f, W): return np.array([ev_classical(f, w) for w in W])


def TIF(Pset, f, W):
    m = ext(f, W)
    T = lower(Pset, m); F = lower(Pset, ~m); I = upper(Pset, m) - lower(Pset, m)
    return T, I, F


results = {}

# ---- Check 1: translation tau preserves values (T = 1 - u(not phi), F = 1 - u(phi), I = u(phi) + u(not phi) - 1)
err = 0.0; nf = 0
for trial in range(400):
    n = rng.choice([2, 3]); W = classical_worlds(n); Pset = rand_credal(len(W), rng.randint(1, 5))
    for _ in range(10):
        f = rand_formula(n, 3); m = ext(f, W)
        T, I, F = TIF(Pset, f, W)
        u = upper(Pset, m); un = upper(Pset, ~m)
        err = max(err, abs(T - (1 - un)), abs(F - (1 - u)), abs(I - (u + un - 1)))
        # a random linear formula and its translation take the same truth value
        a, b, c, alpha = [rng.uniform(-2, 2) for _ in range(4)]
        lhs_np = a * T + b * I + c * F
        lhs_up = -a * un + b * u + b * un - c * u    # constants moved: a(1-un)+b(u+un-1)+c(1-u)
        const = a + (-b) + c
        assert (lhs_np >= alpha) == (lhs_up >= alpha - const) or abs(lhs_np - alpha) < 1e-9
        nf += 1
log("Check 1 (translation tau): formulas", nf, "max error", err)
results["check1_translation"] = dict(formulas=nf, max_error=err)

# ---- Check 2: validities of L_NP and the non-subadditivity of I
viol = dict(I_nonneg=0, I_neg_invariant=0, normalised=0, T_monotone=0, I_union_bound=0, I_true_zero=0)
worst_union = -9
for trial in range(3000):
    n = rng.choice([2, 3]); W = classical_worlds(n); Pset = rand_credal(len(W), rng.randint(1, 4))
    f, g = rand_formula(n, 3), rand_formula(n, 3)
    T, I, F = TIF(Pset, f, W); Tn, In, Fn = TIF(Pset, NOT(f), W)
    if I < -1e-12: viol["I_nonneg"] += 1
    if abs(I - In) > 1e-12: viol["I_neg_invariant"] += 1
    if abs(T + I + F - 1) > 1e-12: viol["normalised"] += 1
    if TIF(Pset, AND(f, g), W)[0] > T + 1e-12: viol["T_monotone"] += 1
    Iu = TIF(Pset, OR(f, g), W)[1]; Ig = TIF(Pset, g, W)[1]; Ia = TIF(Pset, AND(f, g), W)[1]
    worst_union = max(worst_union, Iu - I - Ig - Ia)
    if Iu > I + Ig + Ia + 1e-12: viol["I_union_bound"] += 1
    if abs(TIF(Pset, ("true",), W)[1]) > 1e-12: viol["I_true_zero"] += 1
# explicit counterexample to I(p or q) <= I(p) + I(q): worlds (p,q) in order FF, FT, TF, TT
W2 = classical_worlds(2)
P1 = np.array([0, .5, .5, 0]); P2 = np.array([.5, 0, 0, .5])
ce = dict(I_p=TIF([P1, P2], P, W2)[1], I_q=TIF([P1, P2], Q, W2)[1], I_por=TIF([P1, P2], OR(P, Q), W2)[1],
          I_pand=TIF([P1, P2], AND(P, Q), W2)[1])
log("Check 2 (L_NP validities): 3000 random cases; violations", viol, "max of I(f|g)-I(f)-I(g)-I(f&g)", worst_union)
log("   counterexample to subadditivity:", ce)
results["check2_validities"] = dict(violations=viol, worst_union_slack=worst_union, subadditivity_counterexample=ce)

# ---- Check 3: weak (all normalised dual NPs) versus coherent semantics: the C1 pattern on two atoms
# cells a1 = p&q, a2 = p&~q, a3 = ~p partition the worlds. Coherent: T(ai) >= 0.4 for all i is infeasible.
cells = [AND(P, Q), AND(P, NOT(Q)), NOT(P)]
A = [[-float(ev_classical(c, w)) for w in W2] for c in cells]
r = linprog(np.zeros(4), A_ub=A, b_ub=[-0.4] * 3, A_eq=[np.ones(4)], b_eq=[1], bounds=[(0, 1)] * 4)
coherent_feasible = r.status == 0
# weak model on the 16 events of 2 atoms: T(E) for each event E, T(E)+T(E^c) <= 1, T(Omega)=1, T(empty)=0
events = list(range(16)); masks = {e: [bool(e >> j & 1) for j in range(4)] for e in events}
def ev_index(f): return sum((1 << j) for j, w in enumerate(W2) if ev_classical(f, w))
Aub, bub = [], []
for e in events:
    row = np.zeros(16); row[e] += 1; row[15 - e] += 1; Aub.append(row); bub.append(1)
for c in cells:
    row = np.zeros(16); row[ev_index(c)] = -1; Aub.append(row); bub.append(-0.4)
Aeq = [np.eye(16)[15], np.eye(16)[0]]
r2 = linprog(np.zeros(16), A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=[1, 0], bounds=[(0, 1)] * 16)
log("Check 3 (C1 pattern): coherent feasible =", coherent_feasible, "; weak (normalised dual) feasible =", r2.status == 0)
# C2 pattern: weak model with T(p&q) < T(p&q... ) violating monotonicity: T(p) = .5, T(p or q) = .2
Aub2, bub2 = list(Aub[:16]), list(bub[:16])
Aeq2 = Aeq + [np.eye(16)[ev_index(P)], np.eye(16)[ev_index(OR(P, Q))]]
r3 = linprog(np.zeros(16), A_ub=Aub2, b_ub=bub2, A_eq=Aeq2, b_eq=[1, 0, .5, .2], bounds=[(0, 1)] * 16)
# coherent version of C2: a credal set with lower envelope .5 on p and .2 on p|q needs P with P(p) >= .5 and P(p|q) <= .2
rowp = [-float(ev_classical(P, w)) for w in W2]; rowpq = [float(ev_classical(OR(P, Q), w)) for w in W2]
r4 = linprog(np.zeros(4), A_ub=[rowp, rowpq], b_ub=[-0.5, 0.2], A_eq=[np.ones(4)], b_eq=[1], bounds=[(0, 1)] * 4)
log("   C2 pattern T(p)=.5, T(p|q)=.2: weak feasible =", r3.status == 0, "; coherent (credal LP) feasible =", r4.status == 0)
results["check3_weak_vs_coherent"] = dict(C1_coherent_feasible=bool(coherent_feasible), C1_weak_feasible=bool(r2.status == 0),
                                          C2_weak_feasible=bool(r3.status == 0), C2_coherent_feasible=bool(r4.status == 0))

# ---------------------------------------------------------------- lifted frames
S_MIN = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
S_BEL = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 0)]       # t, f, b, n with E_T={t,b}, E_F={f,b}, E_I={n}
S_DIS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]                   # a, iota, a-bar
FRAMES = dict(S_min=S_MIN, S_Bel=S_BEL, S_dis=S_DIS)


def lifted_worlds(S, n): return list(itertools.product(S, repeat=n))


def lev(f, Wl, X):  # event of component X (0=T,1=I,2=F) as boolean vector over lifted worlds
    return np.array([ev_lift(f, w)[X] for w in Wl])


# ---- Check 5: atomic faithful regions (Theorem 8(a),(d)): x realisable by a credal set on S iff K(x) faithful
def realisable(S, x):
    """x realisable iff for each component k there is P on S with P(E_j) >= x_j for all j and P(E_k) <= x_k."""
    M = np.array([[s[j] for s in S] for j in range(3)], float)
    for k in range(3):
        A = list(-M); b = list(-np.array(x))
        A.append(M[k]); b.append(x[k])
        r = linprog(np.zeros(len(S)), A_ub=A, b_ub=b, A_eq=[np.ones(len(S))], b_eq=[1], bounds=[(0, 1)] * len(S))
        if r.status != 0: return False
    return True

region = dict(S_min=lambda x: True, S_Bel=lambda x: x[0] + x[1] <= 1 + 1e-9 and x[2] + x[1] <= 1 + 1e-9,
              S_dis=lambda x: sum(x) <= 1 + 1e-9)
mism = {k: 0 for k in FRAMES}; npts = 3000
for _ in range(npts):
    x = [rng.random() for _ in range(3)]
    for k, S in FRAMES.items():
        if realisable(S, x) != region[k](x): mism[k] += 1
log("Check 5 (faithful regions, Theorem 8):", npts, "points, mismatches", mism)
results["check5_faithful_regions"] = dict(points=npts, mismatches=mism)


# ---- Check 4: Farkas certificates are integer cover instances (restricted-domain completeness, Theorem 12.6.4)
def check_cover_instance(D, lam, n, k, A0):
    """D: list of boolean event vectors; lam: integer multiplicities. Lower-form cover condition:
    sum_i lam_i 1_{A_i} - n 1_{A0} <= k pointwise."""
    s = sum(l * D[i].astype(int) for i, l in enumerate(lam) if l) if any(lam) else 0
    s = s - n * A0.astype(int)
    return bool(np.all(s <= k))


def certificate(D, x, i0):
    """Maximise sum lam_A x_A - lam0 x_A0 - c subject to sum lam_A 1_A - lam0 1_A0 <= c pointwise, lam>=0,
    normalisation sum lam + lam0 <= 1. Positive optimum = infeasibility certificate (Farkas)."""
    m, nw = len(D), len(D[0])
    # variables: lam (m), lam0, c (free)
    cobj = -np.concatenate([np.array(x), [-x[i0]], [-1.0]])
    A, b = [], []
    for w in range(nw):
        A.append(np.concatenate([D[:, w].astype(float), [-float(D[i0, w])], [-1.0]])); b.append(0.0)
    A.append(np.concatenate([np.ones(m), [1.0], [0.0]])); b.append(1.0)
    r = linprog(cobj, A_ub=A, b_ub=b, bounds=[(0, None)] * (m + 1) + [(None, None)], method="highs-ds")
    return -r.fun, r.x


def lp_feasible(D, x, i0):
    nw = D.shape[1]
    A = list(-D.astype(float)); b = list(-np.array(x)); A.append(D[i0].astype(float)); b.append(x[i0])
    r = linprog(np.zeros(nw), A_ub=A, b_ub=b, A_eq=[np.ones(nw)], b_eq=[1], bounds=[(0, 1)] * nw)
    return r.status == 0, (r.x if r.status == 0 else None)

stats = dict(cases=0, realisable=0, not_realisable=0, certificates_valid=0, certificates_bad=0, envelope_err=0.0,
             sound_violations=0, sound_instances=0)
for trial in range(600):
    name = rng.choice(list(FRAMES)); S = FRAMES[name]; n = rng.choice([1, 2])
    Wl = lifted_worlds(S, n)
    fs = [rand_formula(n, 2) for _ in range(rng.randint(2, 5))]
    D = [np.ones(len(Wl), bool), np.zeros(len(Wl), bool)]           # Omega (true^T) and empty (true^F)
    for f in fs:
        for X in range(3): D.append(lev(f, Wl, X))
    # remove duplicate events (axiom L5_S identifies them)
    uniq = []
    for d in D:
        if not any(np.array_equal(d, u) for u in uniq): uniq.append(d)
    D = np.array(uniq)
    if rng.random() < 0.5:   # a realisable x: lower envelope of a random credal set
        Pset = [nrng.dirichlet(np.ones(len(Wl))) for _ in range(rng.randint(1, 4))]
        x = [min(float(p[d].sum()) for p in Pset) for d in D]
    else:
        x = [rng.random() for _ in D]; x[0] = 1.0; x[1] = 0.0
    stats["cases"] += 1
    feas = [lp_feasible(D, x, i0) for i0 in range(len(D))]
    if all(f for f, _ in feas):
        stats["realisable"] += 1
        Ps = [p for _, p in feas]
        env = [min(float(p[d].sum()) for p in Ps) for d in D]
        stats["envelope_err"] = max(stats["envelope_err"], max(abs(e - xx) for e, xx in zip(env, x)))
    else:
        stats["not_realisable"] += 1
        i0 = next(i for i, (f, _) in enumerate(feas) if not f)
        val, z = certificate(D, x, i0)
        m = len(D)
        # rationalise and scale to integers
        fr = [Fraction(v).limit_denominator(1000) for v in z]
        den = 1
        for q in fr: den = den * q.denominator // np.gcd(den, q.denominator)
        ints = [int(q * den) for q in fr]
        lam, n0, c = ints[:m], ints[m], ints[m + 1]
        if c < 0:                        # shift with |c| copies of Omega (index 0) as in the proof
            lam[0] += -c; c = 0
        ok_cover = check_cover_instance(D, lam, n0, c, D[i0])
        lhs = sum(l * xx for l, xx in zip(lam, x)) - n0 * x[i0]
        if val > 1e-9 and ok_cover and lhs > c + 1e-9: stats["certificates_valid"] += 1
        else: stats["certificates_bad"] += 1
    # soundness: realised x never violates a random integer cover instance
    Pset = [nrng.dirichlet(np.ones(len(Wl))) for _ in range(3)]
    xr = [min(float(p[d].sum()) for p in Pset) for d in D]
    for _ in range(20):
        lam = [rng.randint(0, 2) for _ in D]; i0 = rng.randrange(len(D)); n0 = rng.randint(0, 2)
        s = sum(l * D[i].astype(int) for i, l in enumerate(lam)) - n0 * D[i0].astype(int)
        k = int(max(0, s.max())); stats["sound_instances"] += 1
        if sum(l * xx for l, xx in zip(lam, xr)) - n0 * xr[i0] > k + 1e-9: stats["sound_violations"] += 1
log("Check 4 (restricted-domain completeness via Farkas):", stats)
results["check4_farkas_covers"] = stats

# ---- Check 6: lifted validities
vi = dict(I_same_atoms=0, I_conj_ge_max=0, T_conj_frechet=0, F_conj_ge_max=0, Belnap_region=0)
for trial in range(1500):
    name = rng.choice(list(FRAMES)); S = FRAMES[name]; Wl = lifted_worlds(S, 2)
    Pset = [nrng.dirichlet(np.ones(len(Wl)) * 0.5) for _ in range(rng.randint(1, 4))]
    L = lambda f, X: min(float(p[lev(f, Wl, X)].sum()) for p in Pset)
    if max(abs(L(AND(P, Q), 1) - L(OR(P, Q), 1)), abs(L(AND(P, Q), 1) - L(AND(NOT(P), Q), 1))) > 1e-12: vi["I_same_atoms"] += 1
    if L(AND(P, Q), 1) < max(L(P, 1), L(Q, 1)) - 1e-12: vi["I_conj_ge_max"] += 1
    if L(AND(P, Q), 0) < L(P, 0) + L(Q, 0) - 1 - 1e-12: vi["T_conj_frechet"] += 1
    if L(AND(P, Q), 2) < max(L(P, 2), L(Q, 2)) - 1e-12: vi["F_conj_ge_max"] += 1
    if name == "S_Bel" and (L(P, 0) + L(P, 1) > 1 + 1e-12 or L(P, 2) + L(P, 1) > 1 + 1e-12): vi["Belnap_region"] += 1
# tightness of T(p&q) >= T(p)+T(q)-1 on S_min (Theorem 9(b)): LP minimum over joint measures with atomic bounds
Wl = lifted_worlds(S_MIN, 2); tight_err = 0.0
for _ in range(300):
    x1 = [rng.random() for _ in range(3)]; x2 = [rng.random() for _ in range(3)]
    A, b = [], []
    for X in range(3):
        A.append(-lev(P, Wl, X).astype(float)); b.append(-x1[X]); A.append(-lev(Q, Wl, X).astype(float)); b.append(-x2[X])
    r = linprog(lev(AND(P, Q), Wl, 0).astype(float), A_ub=A, b_ub=b, A_eq=[np.ones(len(Wl))], b_eq=[1], bounds=[(0, 1)] * len(Wl))
    tight_err = max(tight_err, abs(r.fun - max(0, x1[0] + x2[0] - 1)))
log("Check 6 (lifted validities): violations", vi, "; tightness of Frechet bound on S_min, max error", tight_err)
results["check6_lifted_validities"] = dict(violations=vi, frechet_tightness_max_error=tight_err)

# ---- Check 7: T-consequence on the frames is LP (S_min), FDE (S_Bel), K3 (S_dis)
def all_formulas(depth):
    fs = [P, Q]
    for _ in range(depth):
        new = list(fs)
        for f in fs: new.append(NOT(f))
        for f, g in itertools.product(fs, repeat=2): new += [AND(f, g), OR(f, g)]
        # deduplicate by string
        seen = {}; [seen.setdefault(str(h), h) for h in new]; fs = list(seen.values())
    return fs

def three_val(f, v):   # strong Kleene tables on {0, .5, 1}
    t = f[0]
    if t == "atom": return v[f[1]]
    if t == "not": return 1 - three_val(f[1], v)
    a, b = three_val(f[1], v), three_val(f[2], v)
    return min(a, b) if t == "and" else max(a, b)

def fde(f, v):   # Dunn: v(p) subset of {T,F} as pair (told true, told false)
    t = f[0]
    if t == "atom": return v[f[1]]
    if t == "not": a = fde(f[1], v); return (a[1], a[0])
    a, b = fde(f[1], v), fde(f[2], v)
    return (a[0] and b[0], a[1] or b[1]) if t == "and" else (a[0] or b[0], a[1] and b[1])

forms = all_formulas(2); log("   formulas enumerated:", len(forms))
mm = dict(S_min_vs_LP=0, S_Bel_vs_FDE=0, S_dis_vs_K3=0); pairs = 0
V3 = list(itertools.product([0, .5, 1], repeat=2)); V4 = list(itertools.product(itertools.product([False, True], repeat=2), repeat=2))
sub = rng.sample(forms, 120)
for f, g in itertools.product(sub, repeat=2):
    pairs += 1
    cons = {}
    for name, S in FRAMES.items():
        Wl = lifted_worlds(S, 2)
        cons[name] = all((not ev_lift(f, w)[0]) or ev_lift(g, w)[0] for w in Wl)
    lp = all(three_val(f, v) < .5 or three_val(g, v) >= .5 for v in V3)
    k3 = all(three_val(f, v) < 1 or three_val(g, v) == 1 for v in V3)
    fd = all((not fde(f, v)[0]) or fde(g, v)[0] for v in V4)
    mm["S_min_vs_LP"] += cons["S_min"] != lp; mm["S_Bel_vs_FDE"] += cons["S_Bel"] != fd; mm["S_dis_vs_K3"] += cons["S_dis"] != k3
expl = {}
for name, S in FRAMES.items():
    Wl = lifted_worlds(S, 2)
    expl[name] = dict(explosion=all((not ev_lift(AND(P, NOT(P)), w)[0]) or ev_lift(Q, w)[0] for w in Wl),
                      excluded_middle=all(ev_lift(OR(Q, NOT(Q)), w)[0] for w in Wl))
log("Check 7 (T-consequence = LP / FDE / K3):", pairs, "pairs, mismatches", mm, "; explosion / excluded middle:", expl)
results["check7_consequence"] = dict(pairs=pairs, mismatches=mm, examples=expl)

# ---- Check 8: NP-hardness reduction. chi classically satisfiable iff premise^T true and conclusion^T false at some S-world
def rand_cnf(nv, nc):
    cl = []
    for _ in range(nc):
        lits = [(rng.randrange(nv), rng.random() < .5) for _ in range(3)]
        c = None
        for v, neg in lits:
            l = ("atom", v); l = NOT(l) if neg else l
            c = l if c is None else OR(c, l)
        cl.append(c)
    f = cl[0]
    for c in cl[1:]: f = AND(f, c)
    return f
red_mm = {k: 0 for k in FRAMES}; nred = 0
for _ in range(150):
    nv = 3; chi = rand_cnf(nv, rng.randint(2, 9))
    sat = any(ev_classical(chi, w) for w in classical_worlds(nv))
    prem = chi
    for v in range(nv): prem = AND(prem, OR(("atom", v), NOT(("atom", v))))
    concl = AND(("atom", 0), NOT(("atom", 0)))
    for v in range(1, nv): concl = OR(concl, AND(("atom", v), NOT(("atom", v))))
    for name, S in FRAMES.items():
        Wl = lifted_worlds(S, nv)
        nonconseq = any(ev_lift(prem, w)[0] and not ev_lift(concl, w)[0] for w in Wl)
        red_mm[name] += nonconseq != sat
    nred += 1
log("Check 8 (NP-hardness reduction):", nred, "CNFs, mismatches", red_mm)
results["check8_reduction"] = dict(cnfs=nred, mismatches=red_mm)

# ---- Check 10: I is not definable from T and F terms on S_min: Dirac at (1,1,1) versus Dirac at (1,0,1)
WA = [((1, 1, 1),)]; WB = [((1, 0, 1),)]
diff = 0
for _ in range(500):
    f = rand_formula(1, 4)
    for X in (0, 2):
        diff += ev_lift(f, WA[0])[X] != ev_lift(f, WB[0])[X]
log("Check 10 (I not definable on S_min): T/F disagreements over 500 formulas =", diff,
    "; I(p) =", int(ev_lift(P, WA[0])[1]), "versus", int(ev_lift(P, WB[0])[1]))
results["check10_I_not_definable"] = dict(TF_disagreements=diff, I_values=[1, 0])


# ---- Check 9: I-sensitive consequence on S_min: [[f^T]] in [[g^T]] and [[g^I]] in [[f^I]]  iff  f |=_LP g and atoms(g) <= atoms(f)
def atoms(f):
    if f[0] == "atom": return {f[1]}
    if f[0] == "true": return set()
    return set().union(*[atoms(h) for h in f[1:]])
Wl = lifted_worlds(S_MIN, 2); mm9 = 0; n9 = 0; holds9 = 0
for f, g in itertools.product(sub, repeat=2):
    sem = all(((not ev_lift(f, w)[0]) or ev_lift(g, w)[0]) and ((not ev_lift(g, w)[1]) or ev_lift(f, w)[1]) for w in Wl)
    lp = all(three_val(f, v) < .5 or three_val(g, v) >= .5 for v in V3)
    syn = lp and atoms(g) <= atoms(f)
    mm9 += sem != syn; n9 += 1; holds9 += sem
log("Check 9 (I-sensitive consequence on S_min = LP + variable inclusion):", n9, "pairs,", holds9, "valid, mismatches", mm9)
results["check9_I_sensitive_consequence"] = dict(pairs=n9, valid=holds9, mismatches=mm9)


# ---- Check 11: signed liftings (off values). Charges = real masses summing to 1 (no positivity).
# (a) on S_min every triple of R^3 is the exact value of one charge; (b) Farkas certificates are exact-cover instances.
M4 = np.array([[s[j] for s in S_MIN] for j in range(3)] + [[1, 1, 1, 1]], float)
err11 = 0.0
for _ in range(2000):
    x = [rng.uniform(-1, 2) for _ in range(3)]
    mu = np.linalg.solve(M4, np.array(x + [1.0]))
    err11 = max(err11, float(np.abs(M4 @ mu - np.array(x + [1.0])).max()))
def lp_feasible_signed(D, x, i0):
    nw = D.shape[1]
    A = list(-D.astype(float)); b = list(-np.array(x)); A.append(D[i0].astype(float)); b.append(x[i0])
    r = linprog(np.zeros(nw), A_ub=A, b_ub=b, A_eq=[np.ones(nw)], b_eq=[1], bounds=[(None, None)] * nw)
    return r.status == 0, (r.x if r.status == 0 else None)
def certificate_signed(D, x, i0):
    m, nw = len(D), len(D[0])
    cobj = -np.concatenate([np.array(x), [-x[i0]], [-1.0]])
    Aeq = [np.concatenate([D[:, w].astype(float), [-float(D[i0, w])], [-1.0]]) for w in range(nw)]
    r = linprog(cobj, A_ub=[np.concatenate([np.ones(m), [1.0], [0.0]])], b_ub=[1.0], A_eq=Aeq, b_eq=[0.0] * nw,
                bounds=[(0, None)] * (m + 1) + [(None, None)], method="highs-ds")
    return -r.fun, r.x
st11 = dict(cases=0, realisable=0, not_realisable=0, certificates_valid=0, certificates_bad=0, envelope_err=0.0, off_values_realised=0)
for trial in range(400):
    name = rng.choice(list(FRAMES)); S = FRAMES[name]; n = rng.choice([1, 2]); Wl = lifted_worlds(S, n)
    fs = [rand_formula(n, 2) for _ in range(rng.randint(2, 4))]
    D = [np.ones(len(Wl), bool), np.zeros(len(Wl), bool)]
    for f in fs:
        for X in range(3): D.append(lev(f, Wl, X))
    uniq = []
    for d in D:
        if not any(np.array_equal(d, u) for u in uniq): uniq.append(d)
    D = np.array(uniq)
    x = [rng.uniform(-0.5, 1.5) for _ in D]; x[0] = 1.0; x[1] = 0.0
    st11["cases"] += 1
    feas = [lp_feasible_signed(D, x, i0) for i0 in range(len(D))]
    if all(f for f, _ in feas):
        st11["realisable"] += 1
        Ps = [p for _, p in feas]
        env = [min(float(p[d].sum()) for p in Ps) for d in D]
        st11["envelope_err"] = max(st11["envelope_err"], max(abs(e - xx) for e, xx in zip(env, x)))
        st11["off_values_realised"] += any(xx > 1 or xx < 0 for xx in x)
    else:
        st11["not_realisable"] += 1
        i0 = next(i for i, (f, _) in enumerate(feas) if not f)
        val, z = certificate_signed(D, x, i0); m = len(D)
        fr = [Fraction(v).limit_denominator(1000) for v in z]
        den = 1
        for q in fr: den = den * q.denominator // np.gcd(den, q.denominator)
        ints = [int(q * den) for q in fr]; lam, n0, c = ints[:m], ints[m], ints[m + 1]
        if c < 0: lam[0] += -c; c = 0
        s_ = sum(l * D[i].astype(int) for i, l in enumerate(lam)) - n0 * D[i0].astype(int)
        exact = bool(np.all(s_ == c))
        lhs = sum(l * xx for l, xx in zip(lam, x)) - n0 * x[i0]
        if val > 1e-9 and exact and lhs > c + 1e-9: st11["certificates_valid"] += 1
        else: st11["certificates_bad"] += 1
log("Check 11 (signed liftings): exact representation on S_min of 2000 triples in [-1,2]^3, max error", err11, ";", st11)
results["check11_signed"] = dict(S_min_exact_error=err11, **st11)

json.dump(results, open(os.path.join(RES, "completeness_checks.json"), "w"), indent=1, default=float)
open(os.path.join(RES, "completeness_log.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
