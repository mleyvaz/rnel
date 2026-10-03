"""(CRC edition: exercise numbers of the plan of 16 chapters, see the block before main.)
Check every numerical answer of the Solutions Manual (solutions_manual.md) by computation.

For every exercise the script computes the answers from the data of the exercise (closed forms, linear programmes on
finite frames, the rnel library, or the recorded result files of the book), formats each value as printed in the
manual, and asserts that the formatted string appears in the solution of that exercise. Where the book quotes the same
number, the script also asserts agreement with the book.

Usage:  python check_solutions.py            (check; writes check_solutions.out; exit code 1 on any failure)
        python check_solutions.py --print    (print the computed answers, used when writing the manual)
Book edition: the manual (solutions_manual.md, exercises_by_chapter.md) is not distributed with the code. When it is
absent from this folder the script prints every computed answer, formatted as in the manual, plus the book
cross-checks (exit code 1 if a cross-check fails); the plan-16 renumbering table plan16.py is bundled here.
"""
import csv
import itertools
import json
import math
import os
import re
import sys

import numpy as np
from scipy.optimize import linprog, brentq

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
BOOKDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # the book/ folder


def book_result(name, expected_rel):
    """Book edition: a result file of another book script. The fresh copy in the current (output) directory is used
    when present (same regenerate run); otherwise the stored expected copy under book/expected/."""
    p = os.path.join(os.getcwd(), name)
    return p if os.path.exists(p) else os.path.join(BOOKDIR, expected_rel)


CASE = os.path.join(BOOKDIR, "data", "cached", "paracetamol_case")  # CACHED summaries (see data/cached/README.md)
from rnel.neutro_stats import estimate, estimate_sources, decompose, interval_sum, credal_interval  # noqa: E402
from rnel.tuple import Reports  # noqa: E402

ANS = {}          # exercise id -> list of (label, value, format)
BOOKCHK = []      # (description, ok)


def ans(ex, label, value, fmt="%.3f"):
    ANS.setdefault(ex, []).append((label, float(value), fmt))


def book(desc, got, want, tol=1e-9):
    BOOKCHK.append((desc, abs(got - want) <= tol, got, want))


# ---------------------------------------------------------------- finite-frame helpers
def lp(c, A_ub=None, b_ub=None, n=None, bounds=None):
    n = n or len(c)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=[np.ones(n)], b_eq=[1.0], bounds=bounds or [(0, None)] * n, method="highs")
    return res.fun if res.status == 0 else None


def events(n):
    return [frozenset(s) for k in range(1, n) for s in itertools.combinations(range(n), k)]


def ind(A, n):
    return np.array([1.0 if i in A else 0.0 for i in range(n)])


def lower_bounds_system(L, n):
    """L: dict event -> lower bound. Returns A_ub, b_ub for P(A) >= L(A)."""
    A = [-ind(E, n) for E in L]
    b = [-L[E] for E in L]
    return np.array(A), np.array(b)


def nat_ext(L, n, E):
    A, b = lower_bounds_system(L, n)
    lo = lp(ind(E, n), A, b, n)
    hi = lp(-ind(E, n), A, b, n)
    return lo, (-hi if hi is not None else None)


def sure_loss(L, n):
    """Uniform relaxation: min eps s.t. P(A) >= L(A) - eps for all assessed A (variables p, eps)."""
    A = [np.append(-ind(E, n), -1.0) for E in L]
    b = [-L[E] for E in L]
    c = np.zeros(n + 1); c[-1] = 1
    res = linprog(c, A_ub=A, b_ub=b, A_eq=[np.append(np.ones(n), 0)], b_eq=[1], bounds=[(0, None)] * (n + 1), method="highs")
    return res.fun


MIN = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
BEL = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 0)]
DIS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def lift_env(pats, x):
    P = np.array(pats, float)
    out = []
    for k in range(P.shape[1]):
        v = lp(P[:, k], -P.T, -np.array(x, float), len(P))
        if v is None:
            return None
        out.append(v)
    return out


def lift_min_event(pats, x, member):
    P = np.array(pats, float)
    return lp(np.array(member, float), -P.T, -np.array(x, float), len(P))


def product_ne(x1, x2, objective, pats=MIN):
    """Natural extension with no dependence assumption on the product of two liftings; objective(pi, pj) -> value."""
    P = np.array(pats, float); m = len(P)
    pairs = list(itertools.product(range(m), range(m)))
    A, b = [], []
    for k in range(P.shape[1]):
        A.append([-P[i, k] for i, j in pairs]); b.append(-x1[k])
        A.append([-P[j, k] for i, j in pairs]); b.append(-x2[k])
    return lp(np.array([objective(P[i], P[j]) for i, j in pairs]), np.array(A), np.array(b), m * m)


def W(u, v): return max(0.0, u + v - 1)
def M(u, v): return min(u, v)
def Pi(u, v): return u * v


def clayton(t):
    return lambda u, v: 0.0 if u <= 0 or v <= 0 else max(u ** -t + v ** -t - 1, 0) ** (-1 / t)


def gumbel(t):
    return lambda u, v: 0.0 if u <= 0 or v <= 0 else math.exp(-((-math.log(u)) ** t + (-math.log(v)) ** t) ** (1 / t))


def frank(th):
    if th == 0:
        return Pi
    def C(u, v):
        return -math.log(1 + (math.expm1(-th * u) * math.expm1(-th * v)) / math.expm1(-th)) / th
    return C


def survival(C):
    return lambda u, v: u + v - 1 + C(1 - u, 1 - v)


def debye1(x):
    if x == 0:
        return 1.0
    from scipy.integrate import quad
    return quad(lambda t: t / math.expm1(t), 0, x)[0] / x


def frank_tau(th):
    return 1 - 4 / th * (1 - debye1(th))


def nnorm(C, xs):
    T, I, F = xs[0]
    for t, i, f in xs[1:]:
        T, I, F = C(T, t), I + i - C(I, i), F + f - C(F, f)
    return T, I, F


def score(x):
    return (2 + x[0] - x[1] - x[2]) / 3


# ================================================================= Chapter 1
ans("1.1", "sum", 0.6 + 0.3 + 0.2, "%.1f")
ans("1.1", "L", 0.6, "%.1f"); ans("1.1", "U", 1 - 0.2, "%.1f")
ans("1.2", "width", (1 - 0.2) - 0.5, "%.1f"); ans("1.2", "unassigned", 1 - 0.5 - 0.1 - 0.2, "%.1f")
e = lift_env(MIN, (0.6, 0.2, 0.4)); assert np.allclose(e, (0.6, 0.2, 0.4))
e2 = lift_env(MIN, (0.6, 0.0, 0.4))
ans("1.3", "L(E_I) neutrosophic", e[1], "%.1f"); ans("1.3", "L(E_I) intuitionistic", e2[1], "%.1f")
ans("1.4", "T+F", 0.65 + 0.55, "%.2f"); ans("1.4", "delta", (0.65 + 0.55 - 1) / 2, "%.2f")
ans("1.4", "glut", lift_min_event(MIN, (0.65, 0.10, 0.55), [p[0] * p[2] for p in MIN]), "%.2f")
es = estimate_sources({"A": Reports(t=4, c=2), "B": Reports(f=3, c=1, n=2)}, 2.0)
bt = decompose(es.x, "type")
ans("1.5", "lo", es.x.lo); ans("1.5", "hi", es.x.hi); ans("1.5", "IC", bt["C"]); ans("1.5", "IN", bt["N"]); ans("1.5", "IG", bt["G"])
ans("1.6", "mu+nu", 0.6 + 0.55, "%.2f"); ans("1.6", "delta", (0.6 + 0.55 - 1) / 2, "%.3f")
ans("1.6", "glut", lift_min_event(MIN, (0.6, 0.0, 0.55), [p[0] * p[2] for p in MIN]), "%.2f")

# ================================================================= Chapter 2 (Foundations)
# 2.1 Ellsberg: P(R u Y) over b in [0, 2/3]
vals = [1 / 3 + (2 / 3 - b) for b in np.linspace(0, 2 / 3, 1001)]
ans("2.1", "L", min(vals)); ans("2.1", "U", max(vals), "%.0f")
ans("2.2", "buy", min(15 * p - 5 for p in (0.4, 0.7)), "%.1f"); ans("2.2", "sell", max(15 * p - 5 for p in (0.4, 0.7)), "%.1f")
d = sure_loss({frozenset([0]): 0.9, frozenset([1]): 0.6}, 2)
ans("2.3", "delta", d, "%.2f")
LF4 = {frozenset([0]): 0.2, frozenset([1]): 0.3, frozenset([0, 1]): 0.6}
ans("2.4", "U2", nat_ext(LF4, 3, frozenset([1]))[1], "%.1f"); ans("2.4", "U12", nat_ext(LF4, 3, frozenset([0, 1]))[1], "%.0f")
mF5 = {frozenset([0]): 0.3, frozenset([1, 2]): 0.4, frozenset([0, 1, 2]): 0.3}
bel = lambda A: sum(v for B, v in mF5.items() if B <= A)
pl = lambda A: sum(v for B, v in mF5.items() if B & A)
A = frozenset([0, 1])
ans("2.5", "Bel", bel(A), "%.1f"); ans("2.5", "Pl-Bel", pl(A) - bel(A), "%.1f"); ans("2.5", "Bel(c)", bel(frozenset([2])), "%.0f")
C2 = clayton(2.0)
ans("2.7", "W", W(0.3, 0.8), "%.1f"); ans("2.7", "Pi", Pi(0.3, 0.8), "%.2f"); ans("2.7", "M", M(0.3, 0.8), "%.1f")
ans("2.7", "Clayton", C2(0.3, 0.8), "%.4f"); ans("2.7", "survival", survival(C2)(0.3, 0.8), "%.4f")
book("Clayton(2) at (0.6, 0.7) = 0.5117 (F.6)", round(C2(0.6, 0.7), 4), 0.5117)
x, y = (0.7, 0.1, 0.3), (0.4, 0.3, 0.2)
alg = (x[0] * y[0], x[1] + y[1] - x[1] * y[1], x[2] + y[2] - x[2] * y[2])
for k, v in zip("TIF", alg): ans("2.8", "alg " + k, v, "%.2f")
ans("2.8", "minmax T", min(x[0], y[0]), "%.1f"); ans("2.8", "frechet T", max(0, x[0] + y[0] - 1), "%.1f")
a_, b_, c_ = 0.4, 0.9, 0.2
tn, tc = a_ * b_, a_ + b_ - a_ * b_
ans("2.9", "conj", (1 - c_) * tn + c_ * tc); ans("2.9", "disj", (1 - c_) * tc + c_ * tn); ans("2.9", "sum", a_ + b_, "%.1f")
ans("2.9", "half", (a_ + b_) / 2, "%.2f")

# ================================================================= Chapter 3 (Preliminaries)
mE21 = {frozenset([0]): 0.2, frozenset([1]): 0.1, frozenset([0, 1]): 0.3, frozenset([0, 1, 2]): 0.4}
bel = lambda A: sum(v for B, v in mE21.items() if B <= A)
pl = lambda A: sum(v for B, v in mE21.items() if B & A)
ans("3.1", "Pl(a)", pl(frozenset([0])), "%.1f"); ans("3.1", "Bel(ab)", bel(frozenset([0, 1])), "%.1f")
ans("3.1", "Pl(c)", pl(frozenset([2])), "%.1f")
ans("3.1", "IDM1 lo", 6 / 12); ans("3.1", "IDM1 hi", 8 / 12); ans("3.1", "IDM12 lo", 9 / 12, "%.2f"); ans("3.1", "IDM12 hi", 11 / 12)
a_, b_, c_ = 0.6, 0.3, 0.25
tn, tc = a_ * b_, a_ + b_ - a_ * b_
ans("3.2", "conj", (1 - c_) * tn + c_ * tc); ans("3.2", "disj", (1 - c_) * tc + c_ * tn)
x1, x2, c_ = (0.6, 0.2, 0.3), (0.3, 0.4, 0.5), 0.25
def plith(C, x1, x2, c):
    T = (1 - c) * C(x1[0], x2[0]) + c * (x1[0] + x2[0] - C(x1[0], x2[0]))
    I = 0.5 * (x1[1] + x2[1])
    Fv = (1 - c) * (x1[2] + x2[2] - C(x1[2], x2[2])) + c * C(x1[2], x2[2])
    return T, I, Fv
for nm, C in (("Pi", Pi), ("M", M)):
    for k, v in zip("TIF", plith(C, x1, x2, c_)):
        ans("3.2", "%s %s" % (nm, k), v)
ans("3.4", "U", 1 - 0.1, "%.1f"); ans("3.4", "sum", 0.3 + 0.5 + 0.1, "%.1f")
r_, s_, W_ = 8, 2, 2
ans("3.5", "b", r_ / (r_ + s_ + W_)); ans("3.5", "d", s_ / 12); ans("3.5", "u", W_ / 12)
ans("3.5", "proj", (r_ + 0.5 * W_) / 12, "%.2f")
p = dict(t=0.3, f=0.2, b=0.4, n=0.1)
ans("3.6", "ET", p["t"] + p["b"], "%.1f"); ans("3.6", "EF", p["f"] + p["b"], "%.1f"); ans("3.6", "sum", p["t"] + p["f"] + 2 * p["b"], "%.1f")
ans("3.7", "ind lo", 0.6 * 0.5, "%.2f"); ans("3.7", "ind hi", 0.8 * 0.7, "%.2f")
ans("3.7", "fre lo", W(0.6, 0.5), "%.1f"); ans("3.7", "fre hi", M(0.8, 0.7), "%.1f")
ans("3.7", "com lo", M(0.6, 0.5), "%.1f")

# ================================================================= Chapter 4 (Reduction)
T31 = {frozenset([0]): 0.4, frozenset([1]): 0.1, frozenset([2]): 0.2, frozenset([0, 1]): 0.3, frozenset([0, 2]): 0.6, frozenset([1, 2]): 0.3}
ans("4.1", "delta", sure_loss(T31, 3), "%.0f")
ne01 = nat_ext(T31, 3, frozenset([0, 1])); ne2 = nat_ext(T31, 3, frozenset([2]))
ans("4.1", "E01", ne01[0], "%.1f"); ans("4.1", "E2", ne2[0], "%.1f")
ans("4.1", "I01 before", 1 - 0.3 - 0.2, "%.1f"); ans("4.1", "I01 after", 1 - ne01[0] - ne2[0], "%.1f")
m33 = {frozenset([0]): 0.2, frozenset([0, 1]): 0.3, frozenset([1, 2]): 0.1, frozenset([0, 1, 2]): 0.4}
bel = lambda A: sum(v for B, v in m33.items() if B <= A)
Om = frozenset([0, 1, 2])
for A_, nm in ((frozenset([0]), "a"), (frozenset([0, 1]), "ab")):
    ans("4.3", "T " + nm, bel(A_), "%.1f"); ans("4.3", "I " + nm, 1 - bel(A_) - bel(Om - A_), "%.1f")
C1 = {E: (0.4 if len(E) == 1 else 0.3) for E in events(3)}
d_c1 = sure_loss(C1, 3)
ans("4.4", "delta", d_c1, "%.3f"); ans("4.4", "deficit", 1 - 3 * 0.4, "%.1f")
book("C1 delta = 1/15 (Appendix B)", d_c1, 1 / 15, 1e-9)
ans("4.5", "I", 1 - 0.3 - 0.5, "%.1f")
T46 = {frozenset([0]): 0.5, frozenset([1]): 0.2, frozenset([2]): 0.1, frozenset([0, 1]): 0.6, frozenset([0, 2]): 0.7, frozenset([1, 2]): 0.4}
ans("4.6", "delta", sure_loss(T46, 3), "%.0f")
e01 = nat_ext(T46, 3, frozenset([0, 1])); e2_ = nat_ext(T46, 3, frozenset([2]))
ans("4.6", "E01", e01[0], "%.1f"); ans("4.6", "E2", e2_[0], "%.1f")
ans("4.6", "I before", 1 - 0.6 - 0.1, "%.1f"); ans("4.6", "I after", 1 - e01[0] - e2_[0], "%.1f")
ans("4.6", "U2", e2_[1], "%.1f")
R = json.load(open(book_result("t1_reduction_hierarchy.json", "expected/dossier/t1_reduction_hierarchy.json")))["random_normalized_dual"]
ans("4.7", "share coherent n3", 100 * R["3"]["coherent"] / R["3"]["samples"], "%.1f")

# ================================================================= Chapter 5 (Singletons)
def sing_check(tr):
    T = [t[0] for t in tr]; I = [t[1] for t in tr]
    D = 1 - sum(T)
    proper = 0 <= D <= sum(I) + 1e-12
    reach = all(I[i] <= D + 1e-12 and D <= sum(I) - I[i] + 1e-12 for i in range(len(tr)))
    return D, proper, reach
def sing_ne(tr, A):
    T = [t[0] for t in tr]; I = [t[1] for t in tr]
    D = 1 - sum(T); TA = sum(T[i] for i in A); IA = sum(I[i] for i in A); IAc = sum(I[i] for i in range(len(tr)) if i not in A)
    return TA + max(0, D - IAc), TA + min(IA, D)
tr = [(0.3, 0.2, 0.5), (0.2, 0.3, 0.5), (0.1, 0.2, 0.7)]
D, pr, re_ = sing_check(tr); lo_, hi_ = sing_ne(tr, [0, 1])
ans("5.1", "Delta", D, "%.1f"); ans("5.1", "lo12", lo_, "%.1f"); ans("5.1", "hi12", hi_, "%.1f"); ans("5.1", "I12", hi_ - lo_, "%.1f")
L51 = {frozenset([i]): tr[i][0] for i in range(3)}
L51.update({frozenset([j for j in range(3) if j != i]): tr[i][2] * 0 + (1 - (1 - tr[i][2])) for i in range(3)})  # P(not i) >= F_i
lpv = nat_ext(L51, 3, frozenset([0, 1])); assert abs(lpv[0] - lo_) < 1e-9 and abs(lpv[1] - hi_) < 1e-9
tr = [(0.5, 0.4, 0.1), (0.2, 0.1, 0.7), (0.1, 0.1, 0.8)]
D, pr, re_ = sing_check(tr); assert pr and not re_
ans("5.2", "Delta", D, "%.1f"); ans("5.2", "u1new", min(1 - 0.1, 1 - 0.2 - 0.1), "%.1f")
tr = [(0.3, 0.1, 0.6), (0.2, 0.1, 0.7), (0.1, 0.1, 0.8)]
D, pr, re_ = sing_check(tr); assert not pr
ans("5.2", "Delta c", D, "%.1f"); ans("5.2", "sumI c", 0.3, "%.1f"); ans("5.2", "sum upper", sum(1 - t[2] for t in tr), "%.1f")
n_ = [12, 5, 3]; s = 2; N = sum(n_)
ans("5.4", "Delta", s / (N + s)); ans("5.4", "lo", 17 / 22); ans("5.4", "hi", 19 / 22)
n_ = [0, 0, 5]; N = 5
ans("5.5", "upper empty", s / (N + s)); ans("5.5", "lower 3", 5 / 7); ans("5.5", "I", 2 / 7)
n_ = [3, 1]; s = 1
ans("5.6", "Delta", s / (sum(n_) + s), "%.1f"); ans("5.6", "T1", 3 / 5, "%.1f")

# ================================================================= Chapter 6 (Classical frame)
for (T, I, F), nm in (((0.7, 0.4, 0.5), "a"), ((0.3, 0.9, 0.6), "d")):
    ans("6.1", "delta " + nm, max(0, (T + F - 1) / 2), "%.1f" if nm == "a" else "%.0f")
def classical_delta(T, F):
    # min eps: exists p in [0,1] with p >= T - eps and p <= 1 - F + eps
    res = linprog([0, 1], A_ub=[[-1, -1], [1, -1]], b_ub=[-T, 1 - F], bounds=[(0, 1), (0, None)], method="highs")
    return res.x[1]
ans("6.2", "delta overset", classical_delta(1.25, 0.0), "%.2f")
ans("6.2", "upper underset", 1 - 0.3, "%.1f")
ans("6.3", "bet hi", 1 - 0.4, "%.1f")
for T_, I_, F_ in ((0.1, 0.2, 0.3), (0.4, 0.2, 0.6)):
    D_ = T_ - F_
    ans("6.3", "sym lo", (1 + D_ - I_) / 2, "%.1f"); ans("6.3", "sym hi", (1 + D_ + I_) / 2, "%.1f")
ans("6.3", "sym (0,1,1) lo", (1 + (0 - 1) - 1) / 2, "%.1f")      # T = 0, I = 1, F = 1
ans("6.3", "sym (0,0,0)", (1 + 0 - 0) / 2, "%.1f")
ans("6.4", "sym lo",(1 + (0.2 - 0.3) - 0.1) / 2, "%.1f"); ans("6.4", "sym hi", (1 + (0.2 - 0.3) + 0.1) / 2, "%.1f")
ans("6.4", "bet hi", 1 - 0.3, "%.1f")
L65 = {frozenset([0]): 0.6, frozenset([1, 2]): 0.5}  # placeholder replaced below by the 2-event joint check
# 6.5: SM upper 0.5 and a stated lower bound 0.6 for graduation (a subset of "SM passed"): on the two events
# G (graduates) contained in S (SM passed): P(G) >= 0.6 and P(S) <= 0.5 ; frame {G, S\G, not S}
L65 = {frozenset([0]): 0.6, frozenset([2]): 0.5}  # P(G) >= 0.6, P(not S) >= 0.5
ans("6.5", "delta", sure_loss(L65, 3), "%.2f")
book("running example: added triple (0.6,0.1,0.3) incurs sure loss 0.05", sure_loss(L65, 3), 0.05, 1e-9)
ans("6.7", "delta", classical_delta(1.1, 0.3), "%.1f")
ans("6.7", "half", (1.1 + 0.3 - 1) / 2, "%.1f"); ans("6.7", "over", 1.1 - 1, "%.1f")

# ================================================================= Chapter 7 (Subjective Logic)
ans("7.1", "proj", 0.3 + 0.4 * 0.5, "%.1f"); ans("7.1", "U", 0.8, "%.1f")
ans("7.1", "T b", 6 / 10, "%.1f")
r_, q_, W_ = 6, 3, 2
Ts, Fs = r_ / (r_ + W_), q_ / (q_ + W_)
ans("7.2", "Ts", Ts, "%.2f"); ans("7.2", "Fs", Fs, "%.1f"); ans("7.2", "sum", Ts + Fs, "%.2f")
ans("7.2", "lo", r_ / (r_ + q_ + W_)); ans("7.2", "hi", (r_ + W_) / (r_ + q_ + W_)); ans("7.2", "delta", (Ts + Fs - 1) / 2)
T_, F_ = 0.4, 0.3
b = T_ * (1 - F_) / (1 - T_ * F_); u = (1 - T_) * (1 - F_) / (1 - T_ * F_)
ans("7.3", "b", b); ans("7.3", "b+u", b + u)
ans("7.4", "proj", 0.2 + 0.3 * 0.5, "%.2f"); ans("7.4", "U", 0.7, "%.1f")
r_, q_, W_ = 2, 2, 4
Ts = r_ / (r_ + W_)
ans("7.5", "Ts", Ts); ans("7.5", "rho T", r_ / (r_ + q_ + W_), "%.2f"); ans("7.5", "rho I", W_ / (r_ + q_ + W_), "%.1f")
ans("7.5", "chart hi", (r_ + W_) / (r_ + q_ + W_), "%.2f"); ans("7.5", "bet hi", 1 - Ts)

# ================================================================= Chapter 8 (Glut lifting)
x = (0.8, 0.3, 0.6)
ans("8.1", "glut", lift_min_event(MIN, x, [p[0] * p[2] for p in MIN]), "%.1f")
ans("8.1", "T+I", 0.8 + 0.3, "%.1f"); ans("8.1", "sum", sum(x), "%.1f")
assert lift_env(BEL, (0.8, 0.1, 0.6)) is not None and np.allclose(lift_env(BEL, (0.8, 0.1, 0.6)), (0.8, 0.1, 0.6))
def ne_ind(x1, x2):
    return (x1[0] * x2[0], x1[1] + x2[1] - x1[1] * x2[1], x1[2] + x2[2] - x1[2] * x2[2])
x1, x2 = (0.7, 0.2, 0.4), (0.6, 0.3, 0.5)
for k, v in zip("TIF", ne_ind(x1, x2)): ans("8.2", "ind " + k, v, "%.2f")
na = [product_ne(x1, x2, lambda a, b_: a[0] * b_[0]), product_ne(x1, x2, lambda a, b_: max(a[1], b_[1])),
      product_ne(x1, x2, lambda a, b_: max(a[2], b_[2]))]
for k, v in zip("TIF", na): ans("8.2", "na " + k, v, "%.1f")
ans("8.2", "com T", min(x1[0], x2[0]), "%.1f")
x1, x2 = (0.5, 0.3, 0.2), (0.4, 0.4, 0.2)
Talg, Ialg = 0.2, 0.3 + 0.4 - 0.12
ans("8.3", "Ialg", Ialg, "%.2f"); ans("8.3", "credal width", 0.8 * 0.8 - 0.5 * 0.4, "%.2f")
ans("8.3", "sigma", Ialg - (0.8 * 0.8 - 0.5 * 0.4), "%.2f")
ans("8.3", "minmax width", min(0.8, 0.8) - min(0.5, 0.4), "%.1f")
ans("8.5", "sum", 0.2 + 0.3 + 0.4, "%.1f")
x1, x2 = (0.9, 0.4, 0.5), (0.8, 0.3, 0.6)
for k, v in zip("TIF", ne_ind(x1, x2)): ans("8.6", "ind " + k, v, "%.2f")
na = [product_ne(x1, x2, lambda a, b_: a[0] * b_[0]), product_ne(x1, x2, lambda a, b_: max(a[1], b_[1])),
      product_ne(x1, x2, lambda a, b_: max(a[2], b_[2]))]
for k, v in zip("TIF", na): ans("8.6", "na " + k, v, "%.1f")
ans("8.6", "com T", min(0.9, 0.8), "%.1f")

# ================================================================= Chapter 9 (Profiles, open problems)
I = {"a": 0.3 + 0.4, "b": 0.3 + 0.1 + 0.4, "c": 0.1 + 0.4}
for k, v in I.items(): ans("9.1", "I" + k, v, "%.1f")
i_ = (0.5, 0.4, 0.3); t = 0.1
w = dict(ab=0.5 * (i_[0] + i_[1] - i_[2] - t), ac=0.5 * (i_[0] + i_[2] - i_[1] - t), bc=0.5 * (i_[1] + i_[2] - i_[0] - t))
ans("9.2", "wab", w["ab"], "%.2f"); ans("9.2", "wac", w["ac"], "%.2f"); ans("9.2", "wbc", w["bc"], "%.2f")
ans("9.2", "t max", min(i_[0] + i_[1] - i_[2], i_[0] + i_[2] - i_[1], i_[1] + i_[2] - i_[0]), "%.1f")
ans("9.4", "lhs", 0.4 + 0.1 + 0.1, "%.1f"); ans("9.4", "rhs", 2 * 0.4, "%.1f"); ans("9.4", "slack", 0.6 - 0.8, "%.1f")
def shuffle(u, v):
    cells = [(0, 0), (1, 2), (2, 1), (3, 3)]
    s = 0.0
    for i, j in cells:
        # diagonal of cell (i, j): points (i/4 + t, j/4 + t), t in [0, 1/4], mass density 1 per unit of t
        lo_ = max(0.0, 0.0); hi_ = min(0.25, u - i / 4, v - j / 4)
        s += max(0.0, hi_ - lo_)
    return s
ans("9.5", "S14", shuffle(0.25, 0.25), "%.2f"); ans("9.5", "S34", shuffle(0.75, 0.75), "%.2f"); ans("9.5", "S12", shuffle(0.5, 0.5), "%.2f")
sig = 0.25 + 0.25 - shuffle(0.25, 0.25) - shuffle(0.75, 0.75) + shuffle(0.5, 0.5)
ans("9.5", "sigma", sig, "%.2f")
book("Prop 8.3.3 sigma = -1/4", sig, -0.25)
mm = {frozenset([0, 1]): 0.3, frozenset([2]): 0.2, frozenset([0, 1, 2]): 0.5}
strad = lambda A: sum(v for B, v in mm.items() if B & A and B - A)
ans("9.7", "Ia", strad(frozenset([0])), "%.1f"); ans("9.7", "Ic", strad(frozenset([2])), "%.1f")

# ================================================================= Chapter 10 (Choosing an N-norm)
x, y = (0.6, 0.2, 0.2), (0.5, 0.3, 0.2)
for nm, C in (("W", W), ("Pi", Pi), ("M", M)):
    z = nnorm(C, [x, y]); ans("10.1", "score " + nm, score(z), "%.4f")
ans("10.1", "sigma Pi", 0.2 * 0.2 + 0.3 * 0.2, "%.2f")
tau = 0.4
ans("10.2", "clayton", 2 * tau / (1 - tau)); ans("10.2", "gumbel", 1 / (1 - tau))
thf = brentq(lambda t: frank_tau(t) - tau, 0.01, 50); ans("10.2", "frank", thf)
C = clayton(2.0)
sC = 0.4 - C(0.2, 0.2) - 1 + C(0.8, 0.8)
ans("10.3", "C02", C(0.2, 0.2), "%.6f"); ans("10.3", "C08", C(0.8, 0.8), "%.6f"); ans("10.3", "sigma", sC, "%.6f")
G = gumbel(2.0)
sG = 1.4 - G(0.7, 0.7) - 1 + G(0.3, 0.3)
ans("10.3", "G07", G(0.7, 0.7), "%.6f"); ans("10.3", "G03", G(0.3, 0.3), "%.6f"); ans("10.3", "sigmaG", sG, "%.6f")
Jn = json.load(open(book_result("nnorm_choice.json", "expected/added_sections/nnorm_choice.json"), encoding="utf-8"))
cr = {c["pair"]: c for c in Jn["crossings"]}
ans("10.4", "tau23", cr["A2-A3"]["tau"]); ans("10.4", "tau13", cr["A1-A3"]["tau"]); ans("10.4", "tau24", cr["A2-A4"]["tau"]); ans("10.4", "tau12", cr["A1-A2"]["tau"])
x1, x2 = (0.6, 0.3, 0.1), (0.5, 0.2, 0.3)
ans("10.5", "sigma", x1[1] * x2[2] + x2[1] * x1[2], "%.2f")
lo_, hi_ = (1 - x1[2]) * (1 - x2[2]), x1[0] * x2[0]
ans("10.5", "width", lo_ - hi_, "%.2f"); ans("10.5", "Ialg", x1[1] + x2[1] - x1[1] * x2[1], "%.2f")
A2 = [(0.95, 0.05, 0.00), (0.95, 0.00, 0.05), (0.45, 0.25, 0.30)]
zw, zm = nnorm(W, A2), nnorm(M, A2)
for k, v in zip("TIF", zw): ans("10.6", "W " + k, v, "%.2f")
for k, v in zip("TIF", zm): ans("10.6", "M " + k, v, "%.2f")
ans("10.6", "sW", score(zw), "%.4f"); ans("10.6", "sM", score(zm), "%.4f")
book("A2 bracket (0.5667, 0.6333)", score(zw), Jn["bracket"]["A2"][0], 1e-9)
tau = 0.25
ans("10.7", "clayton", 2 * tau / (1 - tau)); ans("10.7", "gumbel", 1 / (1 - tau))
ans("10.7", "frank", brentq(lambda t: frank_tau(t) - tau, 0.01, 50))

# ================================================================= Chapter 11 (Off values)
ans("11.2", "L eps", 0.2, "%.1f"); ans("11.2", "Lpm eps", 0.1, "%.1f")
r_, s_, W_ = 3, -2.5, 2
S_ = r_ + s_ + W_
ans("11.3", "S", S_, "%.1f"); ans("11.3", "T", r_ / S_, "%.1f"); ans("11.3", "I", W_ / S_, "%.1f"); ans("11.3", "F", s_ / S_, "%.1f")
ans("11.3", "sure loss", max(0, (max(0, -r_) + max(0, -s_)) - W_) / S_, "%.1f")
q = np.array([1.3, -0.1, -0.2]); nu = -q[q < 0].sum()
ans("11.4", "nu", nu, "%.1f"); ans("11.4", "max", 1 + nu, "%.1f"); ans("11.4", "tv", np.abs(q).sum(), "%.1f")
# 11.5: delta(q) = min_P max_A |P(A) - Q(A)| for q = (1.2, -0.2) on two atoms: variables p in [0,1], eps
qq = (1.2, -0.2)
res = linprog([0, 1], A_ub=[[1, -1], [-1, -1], [-1, -1], [1, -1]], b_ub=[qq[0], -qq[0], -(1 - qq[1]) + 0, 0],
              bounds=[(0, 1), (0, None)], method="highs")
# |p - 1.2| <= eps and |(1-p) - (-0.2)| <= eps  ->  same constraint; result eps = 0.2 at p = 1
ans("11.5", "delta", 1.2 - 1.0, "%.1f")
assert abs(min(max(abs(p - 1.2), abs((1 - p) + 0.2)) for p in np.linspace(0, 1, 10001)) - 0.2) < 1e-9
r_, s_, W_ = 5, -3, 2
S_ = r_ + s_ + W_
ans("11.6", "S", S_, "%.0f"); ans("11.6", "T", r_ / S_, "%.2f"); ans("11.6", "I", W_ / S_, "%.1f"); ans("11.6", "F", s_ / S_, "%.2f")
ans("11.6", "sure loss", max(0, 3 - W_) / S_, "%.2f")
ans("11.7", "nu", 0.2 + 0.1 + 2 * 0.2 * 0.1, "%.2f")
ans("11.7", "tv", (1 + 2 * 0.2) * (1 + 2 * 0.1), "%.2f")
T1 = T2 = -0.2; e1 = e2 = 0.2
ans("11.8", "alg", T1 * T2, "%.2f"); ans("11.8", "ne", min(T1 * T2, T1 * (1 + e2), (1 + e1) * T2, (1 + e1) * (1 + e2)), "%.2f")

# ================================================================= Chapter 12 (Plithogeny)
x1, x2, c_ = (0.6, 0.3, 0.2), (0.4, 0.1, 0.5), 0.3
def sw_T(C, a, b, c): return c * (a + b) + (1 - 2 * c) * C(a, b)
def sw_F(C, a, b, c): return (1 - c) * (a + b) - (1 - 2 * c) * C(a, b)
ans("12.1", "T ind", sw_T(Pi, 0.6, 0.4, c_)); ans("12.1", "F ind", sw_F(Pi, 0.2, 0.5, c_), "%.2f")
ans("12.1", "T com", sw_T(M, 0.6, 0.4, c_), "%.2f"); ans("12.1", "F com", sw_F(M, 0.2, 0.5, c_), "%.2f")
tna = product_ne(x1, x2, lambda a, b_: (1 - c_) * a[0] * b_[0] + c_ * max(a[0], b_[0]))
fna = product_ne(x1, x2, lambda a, b_: c_ * a[2] * b_[2] + (1 - c_) * max(a[2], b_[2]))
ans("12.1", "T na", tna, "%.1f"); ans("12.1", "F na", fna, "%.2f")
ans("12.1", "spread", abs(1 - 2 * c_) * (M(0.6, 0.4) - W(0.6, 0.4)), "%.2f")
ans("12.1", "mean F", 0.5 * (0.2 + 0.5), "%.2f")
f = lambda a, b: 0.3 * (a + b) + 0.4 * a * b
ans("12.2", "left", f(f(0.2, 0.5), 0.9)); ans("12.2", "right", f(0.2, f(0.5, 0.9)))
ans("12.2", "defect", f(f(0.2, 0.5), 0.9) - f(0.2, f(0.5, 0.9)))
ans("12.2", "on", (1 / 3) * (1 + 0.5) + (1 / 3) * (1 * 0.5)); ans("12.2", "off", (1 / 3) * 0.5)
ans("12.3", "lo", 30 / 52); ans("12.3", "hi", 42 / 52); ans("12.3", "drop lo", 30 / 42); ans("12.3", "drop hi", 32 / 42)
gap = 0.6 * (survival(clayton(2.0))(0.3, 0.6) - clayton(2.0)(0.3, 0.6))
ans("12.4", "gap", gap, "%.4f")
I1, I2 = 0.3, 0.5
ans("12.5", "truth ind", I1 * I2, "%.2f"); ans("12.5", "truth na", W(I1, I2), "%.0f")
ans("12.5", "fals ind", I1 + I2 - I1 * I2, "%.2f"); ans("12.5", "fals na", max(I1, I2), "%.1f"); ans("12.5", "plith", 0.5 * (I1 + I2), "%.1f")
xa, xb = (0.5, I1, 0.5), (0.5, I2, 0.5)
assert abs(product_ne(xa, xb, lambda a, b_: a[1] * b_[1]) - W(I1, I2)) < 1e-9
assert abs(product_ne(xa, xb, lambda a, b_: max(a[1], b_[1])) - max(I1, I2)) < 1e-9
assert abs(product_ne(xa, xb, lambda a, b_: 0.5 * a[1] * b_[1] + 0.5 * max(a[1], b_[1])) - 0.5 * (I1 + I2)) < 1e-9
ps = (0.2, 0.4, 0.9)
ind_ = 0.5 * np.prod(ps) + 0.5 * (1 - np.prod([1 - v for v in ps]))
com_ = 0.5 * min(ps) + 0.5 * max(ps)
ans("12.6", "ind", ind_); ans("12.6", "com", com_, "%.2f"); ans("12.6", "mean", np.mean(ps), "%.1f")
ans("12.7", "atoms min", 9 + 1, "%.0f"); ans("12.7", "atoms product", 4 ** 3, "%.0f"); ans("12.7", "atoms hybrid", 1 + 2 + 3 + 1, "%.0f")

# ================================================================= Chapter 13 (Applications)
x1, x2 = (0.80, 0.10, 0.15), (0.60, 0.30, 0.20)
for c_ in (0, 0.25, 0.5):
    ans("13.1", "Tind %g" % c_, sw_T(Pi, 0.8, 0.6, c_), "%.2f"); ans("13.1", "Tcom %g" % c_, sw_T(M, 0.8, 0.6, c_), "%.2f")
    ans("13.1", "Tna %g" % c_, min(sw_T(W, 0.8, 0.6, c_), sw_T(M, 0.8, 0.6, c_)), "%.2f")
ans("13.1", "F ind 0.25", sw_F(Pi, 0.15, 0.2, 0.25), "%.4f"); ans("13.1", "F half", sw_F(Pi, 0.15, 0.2, 0.5), "%.3f")
for nm, (sat, no, ind_n) in (("East", (80, 40, 30)), ("West", (90, 50, 5))):
    N = sat + no + ind_n
    ans("13.2", nm + " lo", sat / (N + 2)); ans("13.2", nm + " hi", (sat + ind_n + 2) / (N + 2))
    ans("13.2", nm + " drop lo", sat / (sat + no + 2)); ans("13.2", nm + " drop hi", (sat + 2) / (sat + no + 2))
ans("13.2", "Manski lo", 80 / 150); ans("13.2", "Manski hi", 110 / 150)
srcs = [(0.60, 0.25, 0.10), (0.50, 0.20, 0.40), (0.65, 0.10, 0.55)]
ans("13.3", "delta", (0.65 + 0.55 - 1) / 2, "%.1f")
ans("13.3", "glut", lift_min_event(MIN, srcs[2], [p[0] * p[2] for p in MIN]), "%.1f")
eq = [np.mean([s[k] for s in srcs]) for k in range(3)]
wts = np.array([1, 2 / 3, 1 / 3]); wts = wts / wts.sum()
wp = [float(np.dot(wts, [s[k] for s in srcs])) for k in range(3)]
for k, v in zip("TIF", eq): ans("13.3", "eq " + k, v, "%.3f")
ans("13.3", "eq hi", 1 - eq[2], "%.2f"); ans("13.3", "eq width", 1 - eq[2] - eq[0], "%.3f")
for k, v in zip("TIF", wp): ans("13.3", "w " + k, v, "%.3f")
ans("13.3", "w hi", 1 - wp[2], "%.3f")
J4 = {"DE": (0.5, 0.1, 0.2), "SA": (0.6, 0.2, 0.4), "FM": (0.8, 0.0, 0.1), "SM": (0.6, 0.2, 0.3)}
def grad(tr):
    l = {k: v[0] for k, v in tr.items()}; u = {k: 1 - v[2] for k, v in tr.items()}
    pr = lambda d: d["DE"] * d["SA"] * d["FM"] * d["SM"]
    sub = lambda d: min(d["DE"], d["SA"]) * min(d["FM"], d["SM"])
    return dict(na=(max(0, sum(l.values()) - 3), min(u.values())), ind=(pr(l), pr(u)), sub=(sub(l), sub(u)), com=(min(l.values()), min(u.values())))
g = grad(J4)
ans("13.4", "ind lo", g["ind"][0]); ans("13.4", "ind hi", g["ind"][1]); ans("13.4", "sub lo", g["sub"][0], "%.2f")
ans("13.4", "sub hi", g["sub"][1], "%.2f"); ans("13.4", "com lo", g["com"][0], "%.1f"); ans("13.4", "com hi", g["com"][1], "%.1f")
ans("13.4", "na hi", g["na"][1], "%.1f"); ans("13.4", "mean I", np.mean([v[1] for v in J4.values()]), "%.3f")
o = {"o1": (0.85, 0.85, 0.85, 0.60), "o2": (0.75, 0.75, 0.75, 0.75), "o3": (0.95, 0.90, 0.65, 0.90)}
for k, v in o.items():
    ans("13.5", k + " fre lo", max(0, sum(v) - 3), "%.2f"); ans("13.5", k + " ind", np.prod(v), "%.3f"); ans("13.5", k + " mean", np.mean(v), "%.4f")
J6 = {"DE": (0.7, 0.1, 0.0), "SA": (0.6, 0.2, 0.4), "FM": (0.8, 0.0, 0.1), "SM": (0.4, 0.3, 0.5)}
g6 = grad(J6)
ans("13.6", "ind lo", g6["ind"][0], "%.4f"); ans("13.6", "ind hi", g6["ind"][1], "%.2f"); ans("13.6", "sub lo", g6["sub"][0], "%.2f")
ans("13.6", "sub hi", g6["sub"][1], "%.1f"); ans("13.6", "com lo", g6["com"][0], "%.1f"); ans("13.6", "na hi", g6["na"][1], "%.1f")
ans("13.6", "na sum", sum(v[0] for v in J6.values()), "%.1f")
ans("13.7", "sum l", 0.5 + 0.6 + 0.8 + 0.4, "%.1f")

# ================================================================= Chapter 14 (Imprecise probability)
P1, P2 = np.array([0.1, 0.6, 0.3]), np.array([0.5, 0.2, 0.3])
P1b, P2b = np.array([0.5, 0.2, 0.3]), np.array([0.1, 0.3, 0.6])
f = np.array([10, 0, 100.0])
ans("14.1", "seg lo", min(f @ P1b, f @ P2b), "%.0f"); ans("14.1", "seg hi", max(f @ P1b, f @ P2b), "%.0f")
Lb = {E: min(ind(E, 3) @ P1b, ind(E, 3) @ P2b) for E in events(3)}
Ab, bb = lower_bounds_system(Lb, 3)
ans("14.1", "core lo", lp(f, Ab, bb, 3), "%.0f"); ans("14.1", "core hi", -lp(-f, Ab, bb, 3), "%.0f")
# 14.2: Dempster vs generalized Bayes
ans("14.2", "dem lo", 0.2, "%.1f"); ans("14.2", "dem hi", 0.2 + 0.3, "%.1f")
# generalized Bayes: max P(a | {a,b}) over allocations of {b,c} (0.5) and Omega (0.3)
best = max(((0.2 + w2) / (0.2 + w2 + w1)) if (0.2 + w2 + w1) > 0 else 0 for w1 in np.linspace(0, 0.5, 51) for w2 in np.linspace(0, 0.3, 31))
ans("14.2", "gb hi", best, "%.0f")
for k_, cts in ((1, (9, 8, 3)), (2, (18, 16, 6)), (3, (27, 24, 9))):
    N = sum(cts)
    ans("14.3", "I x%d" % k_, 2 / (N + 2))
    ans("14.3", "L1 x%d" % k_, cts[0] / (N + 2)); ans("14.3", "U2 x%d" % k_, (cts[1] + 2) / (N + 2))
m1 = dict(A=0.8, C=0.2); m2 = dict(B=0.8, C=0.2)
K = m1["A"] * m2["B"] + m1["A"] * m2["C"] + m1["C"] * m2["B"]
ans("14.4", "K", K, "%.2f")
pA = m1["A"] ** 2 * m2["C"] / (m1["A"] + m2["C"]) + m1["A"] ** 2 * m2["B"] / (m1["A"] + m2["B"])
pC = m1["C"] * m2["C"] + m1["A"] * m2["C"] ** 2 / (m1["A"] + m2["C"]) + m2["B"] * m1["C"] ** 2 / (m2["B"] + m1["C"])
ans("14.4", "PCR5 A", pA, "%.3f"); ans("14.4", "PCR5 C", pC, "%.3f")
ans("14.4", "DSm Bel C", m1["C"] * m2["C"] + m1["A"] * m2["C"] + m1["C"] * m2["B"], "%.2f")
d = linprog([0, 0, 0, 1], A_ub=[[1, 0, 0, -1], [-1, 0, 0, -1], [0, 1, 0, -1], [0, -1, 0, -1], [0, 0, 1, -1], [0, 0, -1, -1]] * 2,
            b_ub=[0.8, -0.8, 0, 0, 0.2, -0.2, 0, 0, 0.8, -0.8, 0.2, -0.2], A_eq=[[1, 1, 1, 0]], b_eq=[1], bounds=[(0, None)] * 4, method="highs").fun
ans("14.4", "delta", d, "%.1f")
ans("14.5", "Manski lo", 50 / 100, "%.1f"); ans("14.5", "Manski hi", 70 / 100, "%.1f")
ans("14.5", "IDM lo", 50 / 102); ans("14.5", "IDM hi", 72 / 102)
m1 = dict(A=0.6, B=0.4); m2 = dict(A=0.2, B=0.8)
K = m1["A"] * m2["B"] + m1["B"] * m2["A"]
ans("14.6", "K", K, "%.2f"); ans("14.6", "Dem A", m1["A"] * m2["A"] / (1 - K), "%.4f")
pA = m1["A"] * m2["A"] + m1["A"] ** 2 * m2["B"] / (m1["A"] + m2["B"]) + m2["A"] ** 2 * m1["B"] / (m2["A"] + m1["B"])
pB = m1["B"] * m2["B"] + m2["B"] ** 2 * m1["A"] / (m1["A"] + m2["B"]) + m1["B"] ** 2 * m2["A"] / (m2["A"] + m1["B"])
ans("14.6", "PCR5 A", pA, "%.4f"); ans("14.6", "PCR5 B", pB, "%.4f")
ans("14.6", "delta", abs(0.6 - 0.2) / 2, "%.1f")

# ================================================================= Chapter 15 (RNEL, empirical)
ans("15.1", "Kb", min(9, 11) - 3, "%.0f"); ans("15.1", "c", 2 * 6 / 22)
ans("15.3", "V", 4, "%.0f")
srcs = [(6, 1), (2, 5)]
R_, S_ = sum(a for a, b in srcs), sum(b for a, b in srcs)
Kw = sum(min(a, b) for a, b in srcs); Kb = min(R_, S_) - Kw
ans("15.5", "R", R_, "%.0f"); ans("15.5", "S", S_, "%.0f"); ans("15.5", "Kw", Kw, "%.0f"); ans("15.5", "Kb", Kb, "%.0f")
ans("15.5", "c", 2 * Kb / (R_ + S_ + 2))
ans("15.6", "state bytes", 3 * 10 * 8, "%.0f"); ans("15.6", "credal bytes", 500 * 10 * 8, "%.0f"); ans("15.6", "ratio", 500 / 3, "%.1f")
E36 = os.path.join(BOOKDIR, "data", "cached", "experiments", "36_credal_vs_neutro_real_conflict", "results")
sA = json.load(open(E36 + "/summary_averitec.json"))
ans("15.7", "hull", sA["task1_conflict"]["width_hull"]["auroc_mean"]); ans("15.7", "flip", 1 - sA["task1_conflict"]["width_hull"]["auroc_mean"])

# ================================================================= Chapter 16 (Statistics with refined indeterminacy)
r = Reports(t=5, f=3, c=1, v=2, n=1)
e = estimate(r, 2.0); bt = decompose(e.x, "type")
ans("16.1", "L", e.x.lo); ans("16.1", "U", e.x.hi); ans("16.1", "IC", bt["C"]); ans("16.1", "IU", bt["U"]); ans("16.1", "IN", bt["N"])
ans("16.1", "IG", bt["G"]); ans("16.1", "gap", e.x.width)
lo2, hi2 = interval_sum(e.x, e.not_x)
ans("16.2", "lo", lo2); ans("16.2", "hi", hi2); ans("16.2", "sum", e.total.a)
g = estimate(r, 2.0, glut=True)
ans("16.3", "total", g.total.a); ans("16.3", "lo", g.x.lo); ans("16.3", "hi", g.x.hi); ans("16.3", "notx lo", g.not_x.lo)
p_ = estimate(r, 2.0, gap=True)
ans("16.4", "total", p_.total.a); ans("16.4", "lo", p_.x.lo); ans("16.4", "hi", p_.x.hi)
es = estimate_sources({"A": Reports(t=2, c=2), "B": Reports(f=2, v=1)}, 1.0)
bs, bt = decompose(es.x, "source"), decompose(es.x, "type")
ans("16.5", "lo", es.x.lo, "%.3f"); ans("16.5", "hi", es.x.hi, "%.3f"); ans("16.5", "A", bs["A"], "%.3f"); ans("16.5", "B", bs["B"], "%.3f")
ans("16.5", "G", bs["G"], "%.3f"); ans("16.5", "C", bt["C"], "%.3f"); ans("16.5", "U", bt["U"], "%.3f")
summ = json.load(open(CASE + "/results/summary.json", encoding="utf-8"))
fin = summ["final_main"]
eA = estimate(Reports(t=fin["n_t"], f=fin["n_f"], c=fin["n_c"], v=fin["n_v"], extra=(fin["n_A"],)), 2.0)
wA = decompose(eA.x, "type")
ans("16.6", "S", fin["n_t"] + fin["n_f"] + fin["n_c"] + fin["n_v"] + fin["n_A"] + 2, "%.0f")
ans("16.6", "lo", eA.x.lo); ans("16.6", "hi", eA.x.hi); ans("16.6", "IA", wA["I1"]); ans("16.6", "IU", wA["U"]); ans("16.6", "IC", wA["C"])
book("case 2026 lower 0.174 (Section 12.7.7)", round(eA.x.lo, 3), 0.174)
sim = {}
for x in csv.DictReader(open(CASE + "/simulation/results/sim_summary.csv", encoding="utf-8")):
    sim.setdefault(x["pi_conf"], {})[x["policy"]] = float(x["O1_abs_error"])
ans("16.7", "typed 0", sim["0.0"]["typed (ties->IDENT)"], "%.4f"); ans("16.7", "learned 0", sim["0.0"]["interval-learned"], "%.4f")
pol = [k for k in sim["0.0"] if k != "oracle"]
reg = {k: max(sim[pi][k] - min(sim[pi][q] for q in pol) for pi in sim) for k in pol}
ans("16.7", "reg typed", reg["typed (ties->IDENT)"], "%.4f"); ans("16.7", "reg learned", reg["interval-learned"], "%.4f")
ans("16.7", "reg fixed min", min(reg[k] for k in ("always MORE", "always LARGER", "always IDENT")), "%.4f")
cum = [x for x in csv.DictReader(open(CASE + "/results/cumulative.csv", encoding="utf-8")) if x["variant"] == "S1_weak_as_undetermined" and x["year"] == "2026"][0]
eS = estimate(Reports(t=float(cum["n_t"]), f=float(cum["n_f"]), c=float(cum["n_c"]), v=float(cum["n_v"]), extra=(float(cum["n_A"]),)), 2.0)
ans("16.8", "lo", eS.x.lo); ans("16.8", "hi", eS.x.hi); ans("16.8", "IU", decompose(eS.x, "type")["U"])
for k in ("n_t", "n_f", "n_v"):
    ans("16.8", k, float(cum[k]), "%.0f")

# ================================================================= Chapter 17 (Completeness)
ans("17.1", "u(not p)", 1 - 0.4, "%.1f"); ans("17.1", "u sum", 1 + 0.2, "%.1f")
ans("17.2", "sum", 3 * 0.4, "%.1f")
worlds = ["~p~q", "~pq", "p~q", "pq"]
K2 = [np.array([0, 0.5, 0.5, 0]), np.array([0.5, 0, 0, 0.5])]
ev = dict(p=[0, 0, 1, 1], q=[0, 1, 0, 1], por=[0, 1, 1, 1], pand=[0, 0, 0, 1])
Iv = {k: max(P @ np.array(v) for P in K2) - min(P @ np.array(v) for P in K2) for k, v in ev.items()}
ans("17.3", "Ip", Iv["p"], "%.0f"); ans("17.3", "Ipor", Iv["por"], "%.1f"); ans("17.3", "Ipand", Iv["pand"], "%.1f")
ans("17.5", "I111", 1, "%.0f")
ans("17.7", "T(p)", -1 + 2, "%.0f"); ans("17.7", "T(p and not p)", 2, "%.0f")

# ================================================================= Chapter 18 (Critics)
ans("18.1", "delta", (1 + 1 - 1) / 2, "%.1f"); ans("18.1", "glut", 1 + 1 - 1, "%.0f")
ans("18.2", "U", 1 - 0.25, "%.2f"); ans("18.2", "I", 0.75 - 0.4, "%.2f")
es18 = estimate_sources({"A": Reports(t=4, c=2), "B": Reports(f=3, c=1, n=2)}, 2.0)
ans("18.4", "lo", es18.x.lo); ans("18.4", "hi", es18.x.hi); ans("18.4", "sum", es18.total.a)
ans("18.4", "ia lo", interval_sum(es18.x, es18.not_x)[0]); ans("18.4", "ia hi", interval_sum(es18.x, es18.not_x)[1])
ans("18.4", "glut", estimate_sources({"A": Reports(t=4, c=2), "B": Reports(f=3, c=1, n=2)}, 2.0, glut=True).total.a)
ans("18.3", "I18",2 / 20, "%.1f"); ans("18.3", "I36", 2 / 38, "%.4f")
ans("18.5", "c1 lo", 5 / 11); ans("18.5", "c1 hi", 7 / 11); ans("18.5", "c2 hi", 6 / 11)
ans("18.5", "c1 lo x3", 15 / 29); ans("18.5", "c2 hi x3", 14 / 29); ans("18.5", "I", 2 / 11); ans("18.5", "I x3", 2 / 29)

# ================================================================= Chapter 19 (Discussion)
T_, F_, W_ = 0.7, 0.5, 2
r_, q_ = W_ * T_ / (1 - T_), W_ * F_ / (1 - F_)
S_ = r_ + q_ + W_
ans("19.1", "delta", (T_ + F_ - 1) / 2, "%.1f"); ans("19.1", "r", r_); ans("19.1", "q", q_, "%.0f")
ans("19.1", "lo", r_ / S_); ans("19.1", "hi", (r_ + W_) / S_); ans("19.1", "glut", T_ + F_ - 1, "%.1f")
ans("19.3", "sigma", 0.4 * 0.2 + 0.2 * 0.3, "%.2f")
x1, x2 = (0.6, 0.2, 0.3), (0.4, 0.5, 0.1)
ans("19.5", "T", 0.5 * (0.6 + 0.4), "%.1f"); ans("19.5", "I", 0.5 * (0.2 + 0.5), "%.2f"); ans("19.5", "F", 0.5 * (0.3 + 0.1), "%.1f")
for C in (Pi, M, W):
    assert abs(sw_T(C, 0.6, 0.4, 0.5) - 0.5) < 1e-12
f2 = lambda a, b: 0.2 * (a + b) + 0.6 * a * b
ans("19.6", "left", f2(f2(0.1, 0.6), 0.7), "%.4f"); ans("19.6", "right", f2(0.1, f2(0.6, 0.7)), "%.4f")
ans("19.6", "defect", f2(f2(0.1, 0.6), 0.7) - f2(0.1, f2(0.6, 0.7)), "%.3f")
ans("19.6", "formula", 0.2 * 0.8 * (0.7 - 0.1), "%.3f")


# ================================================================= (CRC edition) renumbering to the plan of 16 chapters
# The answers above are computed under the exercise numbers of the 19-chapter plan; plan16.EXMAP renames them.
sys.path.insert(0, HERE)  # book edition: plan16.py is bundled next to this script
from plan16 import EXMAP  # noqa: E402
ANS = {EXMAP[k]: v for k, v in ANS.items()}

# ================================================================= Chapter 14, new exercises (rnel)
from rnel import credal as _credal, conflict as _conflict  # noqa: E402
from rnel import neutro_credal as _nc  # noqa: E402
_t = _credal.to_neutrosophic(0.2, 0.7)
ans("14.3", "T", float(_t[0]), "%.1f"); ans("14.3", "I", float(_t[1]), "%.1f"); ans("14.3", "F", float(_t[2]), "%.1f")
_b = _credal.from_neutrosophic(0.2, 0.5, 0.3); ans("14.3", "L", float(_b[0]), "%.1f"); ans("14.3", "U", float(_b[1]), "%.1f")
_tr = _nc.idm_triple([6, 3, 1], s=2)
for _k in range(3):
    ans("14.4", "T%d" % _k, _tr[_k][0]); ans("14.4", "F%d" % _k, _tr[_k][2])
ans("14.4", "I", _tr[0][1]); ans("14.4", "U1", 1 - _tr[0][2]); ans("14.4", "U2", 1 - _tr[1][2]); ans("14.4", "U3", 1 - _tr[2][2])
assert _nc.interval_dominance(_tr) == [0]
_ne = _nc.natural_extension(_tr, (0, 1)); ans("14.4", "ne T", _ne[0]); ans("14.4", "ne I", _ne[1]); ans("14.4", "ne F", _ne[2])
ans("14.4", "ne U", 1 - _ne[2])
_tr2 = _nc.idm_triple([12, 6, 2], s=2)
for _k in range(3):
    ans("14.4", "2T%d" % _k, _tr2[_k][0]); ans("14.4", "2F%d" % _k, _tr2[_k][2])
ans("14.4", "2I", _tr2[0][1]); assert _nc.interval_dominance(_tr2) == [0]
_prof = [(5, 1), (1, 4), (3, 3)]
_R, _S = sum(r for r, s in _prof), sum(s for r, s in _prof)
ans("14.5", "R", _R, "%.0f"); ans("14.5", "S", _S, "%.0f")
ans("14.5", "Kw", _conflict.k_within(_prof), "%.0f"); ans("14.5", "Kb", _conflict.k_between(_prof), "%.0f")
ans("14.5", "C*", _conflict.c_star(_prof, W=2), "%.3f")
assert abs(_conflict.c_star(list(reversed(_prof)), W=2) - _conflict.c_star(_prof, W=2)) < 1e-12
ans("14.6", "delta", _nc.sure_loss_degree((0.6, 0.3, 0.7)), "%.2f")
_g = _nc.to_glut_frame((0.6, 0.3, 0.7)); assert _g.represents()
ans("14.6", "glut", _g.glut_lower(), "%.1f")
_ep = _g.extreme_points()[0]; ans("14.6", "ep (0,1,1)", _ep[0], "%.1f"); ans("14.6", "ep (1,1,1)", _ep[3], "%.1f")


# ================================================================= check against the manual
def fmt(v, f):
    s = f % v
    if s.startswith("-0") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "−")


def main():
    if "--print" in sys.argv:
        for ex in sorted(ANS, key=lambda e: tuple(int(x) for x in e.split("."))):
            print(ex, "; ".join("%s=%s" % (l, fmt(v, f)) for l, v, f in ANS[ex]))
        for d_, ok, got, want in BOOKCHK:
            print("BOOK", "OK " if ok else "FAIL", d_, got, want)
        return 0
    if not os.path.exists(os.path.join(HERE, "solutions_manual.md")):  # book edition: manual not distributed
        nbad = 0
        for ex in sorted(ANS, key=lambda e: tuple(int(x) for x in e.split("."))):
            for l, v, f in ANS[ex]:
                print("     %-6s %-22s %s" % (ex, l, fmt(v, f)))
        for d_, ok, got, want in BOOKCHK:
            nbad += not ok
            print("%s book   %s (script %s, book %s)" % ("OK  " if ok else "FAIL", d_, got, want))
        print("\nexercises with numerical answers: %d; numerical values: %d; book cross-checks: %d; failures: %d"
              % (len(ANS), sum(len(v) for v in ANS.values()), len(BOOKCHK), nbad))
        return 1 if nbad else 0
    man = open(os.path.join(HERE, "solutions_manual.md"), encoding="utf-8").read()
    ex_md = open(os.path.join(HERE, "exercises_by_chapter.md"), encoding="utf-8").read()
    parts = re.split(r"^#### Solution (\d+\.\d+)\b.*$", man, flags=re.M)
    sol = {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}
    exs = re.findall(r"^\*\*Exercise (\d+\.\d+)\*\*", ex_md, flags=re.M)
    lines, bad = [], 0
    missing_sol = sorted(set(exs) - set(sol), key=lambda e: tuple(map(int, e.split("."))))
    extra_sol = sorted(set(sol) - set(exs), key=lambda e: tuple(map(int, e.split("."))))
    for ex in sorted(ANS, key=lambda e: tuple(int(x) for x in e.split("."))):
        text = sol.get(ex, "").replace("-", "−")
        for l, v, f in ANS[ex]:
            s = fmt(v, f)
            ok = s in text
            bad += not ok
            lines.append("%s %-6s %-22s %s" % ("OK  " if ok else "FAIL", ex, l, s))
    for d_, ok, got, want in BOOKCHK:
        bad += not ok
        lines.append("%s book   %s (script %s, book %s)" % ("OK  " if ok else "FAIL", d_, got, want))
    per = {}
    for e_ in exs:
        per[int(e_.split(".")[0])] = per.get(int(e_.split(".")[0]), 0) + 1
    numeric = sum(len(v) for v in ANS.values())
    lines.append("")
    lines.append("exercises: %d in %d chapters; per chapter %s" % (len(exs), len(per), dict(sorted(per.items()))))
    lines.append("exercises with numerical answers checked: %d; numerical values checked: %d; book cross-checks: %d" %
                 (len(ANS), numeric, len(BOOKCHK)))
    lines.append("exercises without a solution: %s; solutions without an exercise: %s" % (missing_sol or "none", extra_sol or "none"))
    lines.append("failures: %d" % (bad + len(missing_sol) + len(extra_sol)))
    open(os.path.join(HERE, "check_solutions.out"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith("FAIL")))
    print("\n".join(lines[-5:]))
    return 1 if (bad or missing_sol or extra_sol) else 0


if __name__ == "__main__":
    sys.exit(main())
