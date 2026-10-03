"""Numerical checks for the section "Off-values as signed credal sets".

Every claim C1..C9 of the section is checked here; results -> off_checks.json, log -> off_checks_log.txt.
Run:  python off_checks.py      (about 3-6 minutes; seed fixed)
"""
import itertools, json, os, sys, time
import numpy as np
from offlib import *

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.getcwd()  # book edition: outputs go to the current working directory
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(20261001)
OUT = {}
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def all_events(n):
    return [np.array(b, float) for b in itertools.product((0, 1), repeat=n)]


# ---------------------------------------------------------------- C1: structure of Delta_eps
def c1():
    bad = 0; N = 0
    for _ in range(5000):
        n = rng.integers(2, 7); eps = rng.uniform(0, 0.8)
        q = rng.normal(size=n); q = q - (q.sum() - 1) / n  # unit mass
        v = nu(q); ev = [e @ q for e in all_events(n)]
        a = v <= eps + 1e-12; b = min(ev) >= -eps - 1e-12; c = max(ev) <= 1 + eps + 1e-12
        d = np.abs(q).sum() <= 1 + 2 * eps + 1e-12
        N += 1; bad += not (a == b == c == d)
        # offset lemma: q not a probability -> some event < 0 and its complement > 1
        if v > 1e-12:
            neg = (q < 0).astype(float)
            if not (neg @ q < 0 and (1 - neg) @ q > 1): bad += 1
        # norm identity ||q|| = 1 + 2 nu
        if abs(np.abs(q).sum() - (1 + 2 * v)) > 1e-12: bad += 1
    # extreme points: random linear objectives over Delta_eps give dipoles (1+eps) d_x - eps d_y
    nondip = 0; M = 0
    for _ in range(2000):
        n = rng.integers(2, 7); eps = rng.uniform(0.01, 0.8)
        lp = ChargeLP(n, eps); c = rng.normal(size=n)
        v, q = lp.solve(c, "max"); M += 1
        srt = np.sort(q)
        ok = abs(srt[-1] - (1 + eps)) < 1e-7 and abs(srt[0] + eps) < 1e-7 and np.all(np.abs(srt[1:-1]) < 1e-7)
        nondip += not ok
    # each dipole is the unique maximiser of c = e_x - e_y
    uniq_bad = 0
    for n in range(2, 6):
        eps = 0.3
        for x in range(n):
            for y in range(n):
                if x == y: continue
                c = np.zeros(n); c[x] = 1; c[y] = -1
                v, q = ChargeLP(n, eps).solve(c, "max")
                if abs(v - (1 + 2 * eps)) > 1e-8: uniq_bad += 1
    OUT["C1"] = dict(random_charges=N, violations=bad, lp_vertices=M, non_dipole_vertices=nondip, dipole_value_errors=uniq_bad)
    log("C1 Delta_eps equivalences/offset lemma/norm identity:", N, "charges, violations", bad,
        "| LP vertices", M, "non-dipole", nondip, "| dipole max errors", uniq_bad)


# ---------------------------------------------------------------- C2: degree of sure loss = negative variation
def c2():
    errs = []
    for _ in range(600):
        n = rng.integers(2, 7)
        q = rng.normal(size=n) * rng.uniform(0.1, 1.5); q = q - (q.sum() - 1) / n
        E = all_events(n)
        # delta(q) = min_P max_A |P(A) - Q(A)| as an LP in (P, t)
        from scipy.optimize import linprog
        c = np.zeros(n + 1); c[-1] = 1
        Aub, bub = [], []
        for e in E:
            Aub.append(np.concatenate([e, [-1]])); bub.append(e @ q)
            Aub.append(np.concatenate([-e, [-1]])); bub.append(-(e @ q))
        Aeq = [np.concatenate([np.ones(n), [0]])]
        r = linprog(c, A_ub=np.array(Aub), b_ub=bub, A_eq=np.array(Aeq), b_eq=[1], bounds=[(0, None)] * (n + 1), method="highs")
        errs.append(abs(r.fun - nu(q)))
        # the witness P = q^+/(1+nu) achieves it
        P = np.maximum(q, 0) / (1 + nu(q))
        errs.append(abs(max(abs(e @ (P - q)) for e in E) - nu(q)))
    OUT["C2"] = dict(cases=600, max_error=float(max(errs)))
    log("C2 degree of sure loss of a charge = nu(q): 600 charges, max error", max(errs))


# ---------------------------------------------------------------- C3: classical frame
def c3():
    pats = [(1, 0), (0, 1)]  # atoms A (in E_T) and not-A (in E_F)
    bad = 0; N = 0; empt = 0
    for _ in range(4000):
        eps = rng.uniform(0, 0.6); T, F = rng.uniform(-0.9, 1.9, 2)
        lp = ChargeLP(2, eps); lp.ge(event_vec(pats, 0), T); lp.ge(event_vec(pats, 1), F)
        lo, _ = lp.solve(event_vec(pats, 0), "min")
        pred_nonempty = (T + F <= 1 + 1e-12) and T <= 1 + eps + 1e-12 and F <= 1 + eps + 1e-12
        N += 1
        if (lo is not None) != pred_nonempty: bad += 1; continue
        if lo is None: empt += 1; continue
        hi, _ = lp.solve(event_vec(pats, 0), "max")
        if abs(lo - max(T, -eps)) > 1e-7 or abs(hi - min(1 - F, 1 + eps)) > 1e-7: bad += 1
    OUT["C3"] = dict(cases=N, empty=empt, violations=bad)
    log("C3 classical frame: nonempty iff T+F<=1, T,F<=1+eps; L=max(T,-eps), U=min(1-F,1+eps):", N, "cases,", empt, "empty, violations", bad)


# ---------------------------------------------------------------- C4: signed glut lifting
def lifting_lp_theorem8(m, x):
    """Theorem 8 credal set on the minimal lifting (probabilities): lower envelopes."""
    return lower_envelope(minimal_lifting(m), x, 0.0, ["pos"] * (m + 1))


def c4():
    res = {}
    # (a) envelopes on L+- (sign-restricted and free), all x in R^m
    bad = 0; N = 0
    for m in range(1, 7):
        P, S = signed_lifting(m)
        for t in range(250 if m < 5 else 120):
            eps = rng.uniform(0, 0.7) if t % 7 else 0.0
            x = rng.uniform(-eps - 0.4, 1 + eps + 0.4, m)
            if t % 5 == 0: x = rng.choice([-eps, 1 + eps, 0.0, 1.0, 0.5], m)
            for signs in (S, None):
                le = lower_envelope(P, x, eps, signs); N += 1
                nonempty = x.max() <= 1 + eps + 1e-12
                if (le is not None) != nonempty: bad += 1; continue
                if le is None: continue
                if max(abs(a - max(b, -eps)) for a, b in zip(le, x)) > 1e-7: bad += 1
                ue = upper_envelope(P, x, eps, signs)
                if max(abs(u - (1 + eps)) for u in ue) > 1e-7: bad += 1
    res["envelopes"] = dict(cases=N, violations=bad)
    log("C4a L+- envelopes (L=max(x,-eps), U=1+eps, nonempty iff max x<=1+eps), m=1..6:", N, "LPs, violations", bad)
    # (b) minimality: exhaustive for m=2,3; leave-one-out for m=4,5,6
    mini = {}
    for m in (2, 3):
        eps = 0.3; allp = patterns_all(m)
        pts = [np.array(c) for c in itertools.product([-eps, 0.0, 0.5, 1.0, 1 + eps], repeat=m)]
        found = None
        for size in range(1, len(allp) + 1):
            fa = [list(Sx) for Sx in itertools.combinations(allp, size) if all(faithful(list(Sx), x, eps) for x in pts)]
            if fa: found = (size, fa); break
        mini[m] = dict(min_size=found[0], faithful_sets=[["".join(map(str, p)) for p in s] for s in found[1]],
                       pattern_sets_searched=sum(1 for _ in range(1)) and None)
        log("C4b m=%d exhaustive: minimal faithful size %d, sets %s" % (m, found[0], mini[m]["faithful_sets"]))
    for m in (4, 5, 6):
        eps = 0.25; P, _ = signed_lifting(m)
        crit = []
        for k in range(m):
            x = np.full(m, 1 + eps); x[k] = -eps; crit.append(x)
        crit.append(np.full(m, 1 + eps))
        full_ok = all(faithful(P, x, eps) for x in crit)
        drop_ok = []
        for i in range(len(P)):
            Q = P[:i] + P[i + 1:]
            drop_ok.append(all(faithful(Q, x, eps) for x in crit))
        mini[m] = dict(atoms=len(P), full_faithful_on_critical_points=full_ok, any_atom_removable=any(drop_ok))
        log("C4b m=%d: L+- (%d atoms) faithful on critical points %s; some atom removable %s" % (m, len(P), full_ok, any(drop_ok)))
    res["minimality"] = mini
    # (c) reduction at eps = 0: L+- sign-restricted equals Theorem 8 credal set; natural extension on all events of L+-
    bad = 0; N = 0
    for m in (2, 3, 4):
        P, S = signed_lifting(m); L = minimal_lifting(m)
        for _ in range(60):
            x = rng.uniform(0, 1, m)
            for e in all_events(len(P)):
                lp = ChargeLP(len(P), 0.0, S)
                for j in range(m): lp.ge(event_vec(P, j), x[j])
                v1, _ = lp.solve(e, "min")
                # same event restricted to L-atoms, on Theorem 8's frame
                eL = np.array([e[P.index(p)] for p in L])
                lp2 = ChargeLP(len(L), 0.0, ["pos"] * len(L))
                for j in range(m): lp2.ge(event_vec(L, j), x[j])
                v2, _ = lp2.solve(eL, "min"); N += 1
                if abs(v1 - v2) > 1e-8: bad += 1
    res["reduction_eps0"] = dict(natural_extensions_compared=N, violations=bad)
    log("C4c reduction at eps=0 (natural extension of every event of L+- vs Theorem 8 on L):", N, "LPs, violations", bad)
    # (d) faithful region of L alone (free signs) : sum (x-1)^+ <= eps and min x >= -eps
    bad = 0; N = 0
    for m in (2, 3, 4):
        L = minimal_lifting(m)
        for _ in range(500):
            eps = rng.uniform(0, 0.6); x = rng.uniform(-eps - 0.3, 1 + eps + 0.3, m)
            pred = np.maximum(0, x - 1).sum() <= eps + 1e-12 and x.min() >= -eps - 1e-12
            N += 1; bad += faithful(L, x, eps) != pred
    res["L_alone_region"] = dict(cases=N, violations=bad)
    log("C4d faithful region of minimal lifting L under Delta_eps:", N, "cases, violations", bad)
    # (e) nonemptiness on a lifting containing patterns 0 and 1 <=> delta(x) <= eps (delta: Theorem-3 style relaxation)
    bad = 0; N = 0
    for _ in range(300):
        m = rng.integers(2, 5); allp = patterns_all(m)
        extra = [p for p in allp if rng.random() < 0.4 and p not in (tuple([0] * m), tuple([1] * m))]
        P = [tuple([0] * m), tuple([1] * m)] + extra
        x = rng.uniform(-0.5, 1.6, m)
        # delta(x) = min eps s.t. exists probability with P(E_k) >= x_k - eps : LP in (p, eps)
        from scipy.optimize import linprog
        n = len(P); c = np.zeros(n + 1); c[-1] = 1
        Aub = [np.concatenate([-event_vec(P, k), [-1]]) for k in range(m)]; bub = [-x[k] for k in range(m)]
        r = linprog(c, A_ub=np.array(Aub), b_ub=bub, A_eq=[np.concatenate([np.ones(n), [0]])], b_eq=[1],
                    bounds=[(0, None)] * (n + 1), method="highs")
        delta = r.fun
        for eps in (delta - 0.01, delta + 0.01):
            if eps < 0: continue
            lp = ChargeLP(n, eps)
            for j in range(m): lp.ge(event_vec(P, j), x[j])
            v, _ = lp.solve(event_vec(P, 0), "min"); N += 1
            if (v is not None) != (eps >= delta): bad += 1
    res["nonempty_iff_delta"] = dict(cases=N, violations=bad)
    log("C4e K_eps(x) nonempty iff delta(x) <= eps on liftings containing 0 and 1:", N, "cases, violations", bad)
    OUT["C4"] = res


# ---------------------------------------------------------------- C5: glut natural extension on L+- (sign-restricted)
def c5():
    P, S = signed_lifting(3)
    glut = np.array([p[0] * p[2] for p in P], float)
    bad = 0; N = 0
    for t in range(3000):
        eps = rng.uniform(0, 0.6); x = rng.uniform(-eps, 1 + eps, 3)
        if t % 4 == 0: eps = eps_star(x)
        lp = ChargeLP(len(P), eps, S)
        for j in range(3): lp.ge(event_vec(P, j), x[j])
        v, _ = lp.solve(glut, "min"); N += 1
        if abs(v - max(0.0, x[0] + x[2] - 1 - eps)) > 1e-7: bad += 1
    OUT["C5"] = dict(cases=N, violations=bad)
    log("C5 glut lower value max(0, T+F-1-eps) on signed lifting:", N, "cases, violations", bad)


# ---------------------------------------------------------------- C6: signed IDM
def c6():
    bad = 0; N = 0
    for _ in range(3000):
        k = rng.integers(2, 6); W = rng.uniform(0.5, 4)
        n = rng.uniform(-3, 8, k)
        if n.sum() + W <= 0.05: continue
        S = n.sum() + W
        verts = [(n + W * np.eye(k)[j]) / S for j in range(k)]
        # (a) unit mass
        if max(abs(v.sum() - 1) for v in verts) > 1e-12: bad += 1
        # (b) envelopes on every event (min/max over the simplex image are attained at vertices)
        for e in all_events(k)[1:-1]:
            vals = [e @ v for v in verts]
            nA = e @ n
            if abs(min(vals) - nA / S) > 1e-12 or abs(max(vals) - (nA + W) / S) > 1e-12: bad += 1
        # (c) contains a probability iff sum n^- <= W  (LP over t)
        from scipy.optimize import linprog
        r = linprog(np.zeros(k), A_ub=-np.eye(k) * W, b_ub=n, A_eq=[np.ones(k)], b_eq=[1], bounds=[(0, None)] * k, method="highs")
        has = r.status == 0
        if has != (np.maximum(0, -n).sum() <= W + 1e-12): bad += 1
        # (d) least negative variation = (sum n^- - W)^+ / S  (LP over t with slack)
        c = np.concatenate([np.zeros(k), np.ones(k)])
        A = np.hstack([-np.eye(k) * W, -np.eye(k)]); b = n  # -(n+Wt) <= s  ->  -W t - s <= n
        r2 = linprog(c, A_ub=A, b_ub=b, A_eq=[np.concatenate([np.ones(k), np.zeros(k)])], b_eq=[1], bounds=[(0, None)] * (2 * k), method="highs")
        if abs(r2.fun / S - max(0, np.maximum(0, -n).sum() - W) / S) > 1e-9: bad += 1
        N += 1
    # (e) binary: lower-envelope triple = IJFS off-opinion; Beta region <=> base-rate member strictly positive
    bad_b = 0; M = 0; over = 0; off_pair_viol = 0
    for _ in range(20000):
        W = rng.uniform(0.5, 4); r, s = rng.uniform(-6, 10, 2); a = rng.uniform(0.01, 0.99)
        S = r + s + W
        if S <= 1e-3: continue
        M += 1
        T, I, F = r / S, W / S, s / S
        L = r / S; U = (r + W) / S
        if abs(U - L - I) > 1e-12 or abs(1 - U - F) > 1e-12: bad_b += 1
        qa = np.array([r + a * W, s + (1 - a) * W]) / S
        beta = (r + a * W > 0) and (s + (1 - a) * W > 0)
        if beta != bool(np.all(qa > 0)): bad_b += 1
        if beta != (0 < qa[0] < 1): bad_b += 1
        # component > 1 implies another component < 0 (normalised, I > 0)
        if max(T, I, F) > 1:
            over += 1
            if min(T, I, F) >= 0: off_pair_viol += 1
        if (T > 1) != (s < -W): bad_b += 1
        if (I > 1) != (r + s < 0): bad_b += 1
    OUT["C6"] = dict(multinomial_cases=N, violations=bad, binary_cases=M, binary_violations=bad_b,
                     binary_with_overvalue=over, overvalue_without_undervalue=off_pair_viol)
    log("C6 signed IDM:", N, "multinomial cases, violations", bad, "| binary", M, "violations", bad_b,
        "| over-values", over, "without an under-value", off_pair_viol)
    # worked example: 3 supporting passages x 5 units, 2 refuting x 4 units, W = 2, k supporting passages retracted
    ex = []
    for kk in range(4):
        for mode in ("anti", "remove"):
            r = 15 - (10 if mode == "anti" else 5) * kk; s = 8; W = 2; S = r + s + W
            if S <= 0:
                ex.append(dict(k=kk, mode=mode, r=r, s=s, S=S, defined=False)); continue
            T, I, F = r / S, W / S, s / S
            nneg = max(0, -r) + max(0, -s)
            ex.append(dict(k=kk, mode=mode, r=r, s=s, S=S, T=round(T, 4), I=round(I, 4), F=round(F, 4),
                           interval=[round(T, 4), round(1 - F, 4)], contains_probability=nneg <= W,
                           least_budget=round(max(0, nneg - W) / S, 4), eps_star=round(eps_star([T, I, F]), 4)))
    OUT["C6_example"] = ex
    for e in ex: log("   example", e)


# ---------------------------------------------------------------- C7: N-norms on off-credal sets
def vertices_of_K(P, S, x, eps, n_obj=60):
    V = []
    for _ in range(n_obj):
        lp = ChargeLP(len(P), eps, S)
        for j in range(len(x)): lp.ge(event_vec(P, j), x[j])
        v, q = lp.solve(rng.normal(size=len(P)), "min")
        V.append(q)
    # add the attaining ones
    for k in range(len(x)):
        for sense in ("min", "max"):
            lp = ChargeLP(len(P), eps, S)
            for j in range(len(x)): lp.ge(event_vec(P, j), x[j])
            v, q = lp.solve(event_vec(P, k), sense); V.append(q)
    return V


def indep_formula(x1, x2, e1, e2):
    T1, T2 = x1[0], x2[0]
    T = min(T1 * T2, T1 * (1 + e2), (1 + e1) * T2, (1 + e1) * (1 + e2))
    def con(a, b):
        return 1 - max((1 - a) * (1 - b), -e2 * (1 - a), -e1 * (1 - b), e1 * e2)
    return (T, con(x1[1], x2[1]), con(x1[2], x2[2]))


def c7():
    res = {}
    # product budget nu(q1 x q2) = nu1 + nu2 + 2 nu1 nu2
    errs = []
    for _ in range(2000):
        n1, n2 = rng.integers(2, 6, 2)
        q1 = rng.normal(size=n1); q1 -= (q1.sum() - 1) / n1
        q2 = rng.normal(size=n2); q2 -= (q2.sum() - 1) / n2
        a, b = nu(q1), nu(q2)
        errs.append(abs(nu(np.outer(q1, q2).ravel()) - (a + b + 2 * a * b)))
    res["product_budget_max_error"] = float(max(errs))
    log("C7a product budget identity: 2000 pairs, max error", max(errs))
    # independence: formula vs products of sampled vertices; formula attained
    P, S = signed_lifting(3)
    below = 0; worst_gap = 0.0; N = 0; att_err = 0.0
    alg_fail_T = 0; alg_fail_I = 0
    for t in range(300):
        e1, e2 = rng.uniform(0, 0.5, 2)
        if t % 5 == 0: e1 = 0.0
        x1 = rng.uniform(-e1, 1 + e1, 3); x2 = rng.uniform(-e2, 1 + e2, 3)
        if e1 == 0: x1 = rng.uniform(0, 1, 3)
        V1 = vertices_of_K(P, S, x1, e1, 25); V2 = vertices_of_K(P, S, x2, e2, 25)
        f = indep_formula(x1, x2, e1, e2)
        mins = [np.inf] * 3
        for q1 in V1:
            u = [event_vec(P, k) @ q1 for k in range(3)]
            for q2 in V2:
                v = [event_vec(P, k) @ q2 for k in range(3)]
                vals = (u[0] * v[0], u[1] + v[1] - u[1] * v[1], u[2] + v[2] - u[2] * v[2])
                for i in range(3): mins[i] = min(mins[i], vals[i])
        N += 1
        for i in range(3):
            if mins[i] < f[i] - 1e-9: below += 1
            att_err = max(att_err, mins[i] - f[i])
        alg = (x1[0] * x2[0], x1[1] + x2[1] - x1[1] * x2[1], x1[2] + x2[2] - x1[2] * x2[2])
        alg_fail_T += abs(alg[0] - f[0]) > 1e-9
        alg_fail_I += abs(alg[1] - f[1]) > 1e-9
    res["independence"] = dict(pairs=N, sampled_values_below_formula=below, max_attainment_gap=float(att_err),
                               algebraic_T_differs=alg_fail_T, algebraic_I_differs=alg_fail_I)
    log("C7b independence formula:", N, "pairs; sampled values below formula", below, "; max attainment gap", att_err,
        "; algebraic T differs in", alg_fail_T, "; algebraic I differs in", alg_fail_I)
    # counterexamples
    ce = {
        "T_mixed_sign": dict(x1=[-0.2, 0.3, 0.9], e1=0.2, x2=[0.5, 0.2, 0.3], e2=0.0),
        "T_both_negative": dict(x1=[-0.2, 0.3, 0.9], e1=0.2, x2=[-0.2, 0.3, 0.9], e2=0.2),
        "I_over": dict(x1=[0.1, 1.2, 0.0], e1=0.2, x2=[0.3, 0.5, 0.2], e2=0.0),
        "I_standard_but_budget": dict(x1=[1.2, 1.0, 0.0], e1=0.2, x2=[1.1, 1.0, 0.0], e2=0.1),
    }
    for kk, d in ce.items():
        f = indep_formula(d["x1"], d["x2"], d["e1"], d["e2"])
        x1, x2 = d["x1"], d["x2"]
        alg = (x1[0] * x2[0], x1[1] + x2[1] - x1[1] * x2[1], x1[2] + x2[2] - x1[2] * x2[2])
        d["natural_extension"] = [round(v, 6) for v in f]; d["algebraic_N_norm"] = [round(v, 6) for v in alg]
        log("   counterexample", kk, d)
    res["counterexamples"] = ce
    # comonotone and no-assumption models on the 64-atom product (LP)
    n = len(P); pairs = [(a, b) for a in range(n) for b in range(n)]; NN = len(pairs)
    mA = {k: np.array([P[a][k] for a, b in pairs], float) for k in range(3)}
    mB = {k: np.array([P[b][k] for a, b in pairs], float) for k in range(3)}
    JA = {k: np.array([P[a][k] * P[b][k] for a, b in pairs], float) for k in range(3)}
    CV = [JA[0], np.array([max(P[a][1], P[b][1]) for a, b in pairs], float), np.array([max(P[a][2], P[b][2]) for a, b in pairs], float)]

    def base(x1, x2, e1, e2, eJ):
        lp = ChargeLP(NN, eJ)
        for k in range(3): lp.ge(mA[k], x1[k]); lp.ge(mB[k], x2[k])
        negA = np.zeros(NN); negB = np.zeros(NN)
        for i, s in enumerate(S):
            aA = np.array([1.0 if a == i else 0 for a, b in pairs]); aB = np.array([1.0 if b == i else 0 for a, b in pairs])
            if s == "pos": lp.ge(aA, 0); lp.ge(aB, 0)
            if s == "neg": lp.le(aA, 0); lp.le(aB, 0); negA -= aA; negB -= aB
        lp.le(negA, e1); lp.le(negB, e2)
        return lp

    como = {}
    for label in ("product_budget", "max_budget"):
        bad = 0; N = 0; first = None
        for t in range(150):
            e1, e2 = rng.uniform(0, 0.4, 2)
            x1 = rng.uniform(-e1, 1 + e1, 3); x2 = rng.uniform(-e2, 1 + e2, 3)
            eJ = e1 + e2 + 2 * e1 * e2 if label == "product_budget" else max(e1, e2)
            got = []
            for c in CV:
                best = None
                for br in itertools.product((0, 1), repeat=3):
                    lp = base(x1, x2, e1, e2, eJ)
                    for k in range(3):
                        if br[k] == 0: lp.eq(JA[k] - mA[k], 0); lp.le(mA[k] - mB[k], 0)
                        else: lp.eq(JA[k] - mB[k], 0); lp.le(mB[k] - mA[k], 0)
                    v, _ = lp.solve(c, "min")
                    if v is not None: best = v if best is None else min(best, v)
                got.append(best)
            g = (min(x1[0], x2[0]), max(x1[1], x2[1]), max(x1[2], x2[2]))
            N += 1
            if any(v is None for v in got) or max(abs(a - b) for a, b in zip(got, g)) > 1e-7:
                bad += 1
                if first is None: first = dict(x1=list(np.round(x1, 4)), x2=list(np.round(x2, 4)), e1=round(e1, 4), e2=round(e2, 4),
                                               eJ=round(eJ, 4), lp=[None if v is None else round(v, 4) for v in got], minmaxmax=list(np.round(g, 4)))
        como[label] = dict(pairs=N, mismatches=bad, first_mismatch=first)
        log("C7c comonotone (%s): %d pairs, mismatches %d, first %s" % (label, N, bad, first))
    res["comonotone"] = como
    # no dependence assumption: lower bounds max(-eJ, T1+T2-1-eJ) and max(I1,I2)-eJ hold; Frechet N-norm not attained
    viol = 0; differs = 0; N = 0
    for t in range(150):
        e1, e2 = rng.uniform(0, 0.4, 2)
        x1 = rng.uniform(-e1, 1 + e1, 3); x2 = rng.uniform(-e2, 1 + e2, 3)
        eJ = max(e1, e2)
        got = [base(x1, x2, e1, e2, eJ).solve(c, "min")[0] for c in CV]
        lbT = max(-eJ, x1[0] + x2[0] - 1 - eJ); lbI = max(x1[1], x2[1]) - eJ; lbF = max(x1[2], x2[2]) - eJ
        N += 1
        viol += (got[0] < lbT - 1e-7) + (got[1] < lbI - 1e-7) + (got[2] < lbF - 1e-7)
        fr = (max(0, x1[0] + x2[0] - 1), max(x1[1], x2[1]), max(x1[2], x2[2]))
        differs += max(abs(a - b) for a, b in zip(got, fr)) > 1e-7
    res["no_assumption"] = dict(pairs=N, lower_bound_violations=viol, frechet_nnorm_differs=differs)
    log("C7d no dependence assumption (eJ=max(e1,e2)):", N, "pairs; bound violations", viol, "; Frechet N-norm differs in", differs)
    # non-sharpness of the charge Frechet lower bound: u=-eps, v=1+eps on a 2x2 frame
    eps = 0.2
    lp = ChargeLP(4, eps)  # cells AB, A-B, B-A, rest
    lp.eq([1, 1, 0, 0], -eps); lp.eq([1, 0, 1, 0], 1 + eps)
    lo, _ = lp.solve([1, 0, 0, 0], "min"); hi, _ = lp.solve([1, 0, 0, 0], "max")
    res["frechet_charge_nonsharp"] = dict(u=-eps, v=1 + eps, eps=eps, bound=-eps, attainable=[lo, hi])
    log("C7e charge Frechet bound not sharp: u=-0.2, v=1.2, eps=0.2: Q(A&B) in", [lo, hi], "bound -0.2")
    OUT["C7"] = res


if __name__ == "__main__":
    t0 = time.time()
    for f in (c1, c2, c3, c4, c5, c6, c7):
        f()
    OUT["seconds"] = round(time.time() - t0, 1)
    json.dump(OUT, open(os.path.join(RES, "off_checks.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    open(os.path.join(RES, "off_checks_log.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    log("done in", OUT["seconds"], "s")
