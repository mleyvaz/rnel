"""Deterministic worked examples quoted in the section; -> off_examples.json (current directory)"""
import itertools, json, os
import numpy as np
from offlib import *

RES = os.getcwd()  # book edition: outputs go to the current working directory
OUT = {}
P, S = signed_lifting(3)
names = ["".join(map(str, p)) for p in P]

# Example 1: x = (1.2, 0.3, -0.1)
x = np.array([1.2, 0.3, -0.1]); e = eps_star(x)
ex1 = dict(x=x.tolist(), eps_star=e, lower=lower_envelope(P, x, e, S), upper=upper_envelope(P, x, e, S))
# the two witnesses of the proof of Theorem 8.4.2(a)
def witness(x, k, eps):
    m = len(x); t = max(x[k], -eps)
    O = max([max(0.0, x[j] - 1) for j in range(m) if j != k] + [0.0])
    q = dict()
    one = tuple([1] * m); zero = tuple([0] * m)
    ek = tuple(int(j == k) for j in range(m)); nk = tuple(1 - int(j == k) for j in range(m))
    if t >= 0:
        nu_ = max(O, max(0.0, t - 1)); a, b, g, h = t, 1 + nu_ - t, nu_, 0.0
    else:
        nu_ = max(O, -t); a, b, g, h = 0.0, 1 + nu_, nu_ + t, -t
    vec = np.zeros(len(P))
    for pat, val in ((one, a), (nk, b), (zero, -g), (ek, -h)):
        vec[P.index(pat)] += val
    return vec
ws = {}
for k, lab in enumerate("TIF"):
    q = witness(x, k, e)
    ws[lab] = dict(q={names[i]: round(float(q[i]), 6) for i in range(len(P)) if abs(q[i]) > 1e-12}, nu=nu(q),
                   QE=[round(float(event_vec(P, j) @ q), 6) for j in range(3)])
ex1["witnesses"] = ws
glut = np.array([p[0] * p[2] for p in P], float)
lp = ChargeLP(len(P), e, S)
for j in range(3): lp.ge(event_vec(P, j), x[j])
ex1["glut_lower"] = lp.solve(glut, "min")[0]
OUT["example_theorem"] = ex1

# Example 2: classical frame with an over-value T = 1.2
pats = [(1, 0), (0, 1)]
for T, F in ((1.2, -0.3), (1.2, 0.0), (-0.1, 0.5)):
    for eps in (0.0, 0.2, 0.3):
        lp = ChargeLP(2, eps); lp.ge(event_vec(pats, 0), T); lp.ge(event_vec(pats, 1), F)
        lo, _ = lp.solve(event_vec(pats, 0), "min")
        hi = lp.solve(event_vec(pats, 0), "max")[0] if lo is not None else None
        OUT.setdefault("classical", []).append(dict(T=T, F=F, eps=eps, L=lo, U=hi))

# Example 3: no dependence assumption, explicit
n = len(P); pairs = [(a, b) for a in range(n) for b in range(n)]; NN = len(pairs)
mA = {k: np.array([P[a][k] for a, b in pairs], float) for k in range(3)}
mB = {k: np.array([P[b][k] for a, b in pairs], float) for k in range(3)}
CV = [np.array([P[a][0] * P[b][0] for a, b in pairs], float), np.array([max(P[a][1], P[b][1]) for a, b in pairs], float),
      np.array([max(P[a][2], P[b][2]) for a, b in pairs], float)]
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
x1, x2, e1, e2 = [1.2, 0.3, 0.1], [0.6, 0.5, 0.2], 0.2, 0.0
for eJ in (0.0, 0.2):
    if eJ < max(e1, e2):
        OUT.setdefault("no_assumption", []).append(dict(eJ=eJ, note="empty: joint budget below marginal budget")); continue
    got = [base(x1, x2, e1, e2, eJ).solve(c, "min")[0] for c in CV]
    OUT.setdefault("no_assumption", []).append(dict(x1=x1, x2=x2, e1=e1, e2=e2, eJ=eJ, natural_extension=[round(v, 6) for v in got],
                                                    frechet_nnorm=[max(0, x1[0] + x2[0] - 1), max(x1[1], x2[1]), max(x1[2], x2[2])],
                                                    bounds=[max(-eJ, x1[0] + x2[0] - 1 - eJ), max(x1[1], x2[1]) - eJ, max(x1[2], x2[2]) - eJ]))
# standard inputs, eJ = 0: Theorem 9(b)
xs1, xs2 = [0.7, 0.2, 0.3], [0.6, 0.5, 0.2]
got = [base(xs1, xs2, 0, 0, 0.0).solve(c, "min")[0] for c in CV]
OUT["no_assumption_standard_eJ0"] = dict(natural_extension=[round(v, 6) for v in got], frechet=[0.3, 0.5, 0.3])
got = [base(xs1, xs2, 0, 0, 0.1).solve(c, "min")[0] for c in CV]
OUT["no_assumption_standard_eJ01"] = dict(natural_extension=[round(v, 6) for v in got])

# Example 4: signed IDM in the diffident case (Smets): s = 0, r = -1, W = 2
r, s, W = -1.0, 0.0, 2.0; Sg = r + s + W
OUT["diffident"] = dict(r=r, s=s, W=W, T=r / Sg, I=W / Sg, F=s / Sg, interval=[r / Sg, 1.0], w=W / (r + W))
json.dump(OUT, open(os.path.join(RES, "off_examples.json"), "w"), indent=1)
print(json.dumps(OUT, indent=1))
