"""All numbers used in the worked examples and exercises added in the book (blue+yellow text).
Writes foundations_examples.json (current directory) and prints it. Run: python foundations_examples.py (a few seconds)."""
import numpy as np, json
from fractions import Fraction as Fr
from scipy.optimize import linprog

R = {}


def lp_bounds(n, cons, target):
    """cons: list of (event_indices, lo, hi). (min, max) of P(target) over the credal set, or None if empty."""
    A_ub, b_ub = [], []
    for ev, lo, hi in cons:
        row = [1.0 if i in ev else 0.0 for i in range(n)]
        A_ub.append([-x for x in row]); b_ub.append(-lo)
        A_ub.append(row); b_ub.append(hi)
    c = np.array([1.0 if i in target else 0.0 for i in range(n)])
    out = []
    for s in (1, -1):
        r = linprog(s * c, A_ub=A_ub, b_ub=b_ub, A_eq=[[1] * n], b_eq=[1], bounds=[(0, 1)] * n, method='highs')
        if r.status != 0:
            return None
        out.append(s * r.fun)
    return tuple(round(v, 6) for v in out)


def sure_loss(n, cons):
    """Degree of sure loss: least eps with lo - eps <= P(ev) <= hi + eps for all assessed events."""
    A_ub, b_ub = [], []
    for ev, lo, hi in cons:
        row = [1.0 if i in ev else 0.0 for i in range(n)]
        A_ub.append([-x for x in row] + [-1]); b_ub.append(-lo)
        A_ub.append(row + [-1]); b_ub.append(hi)
    r = linprog([0] * n + [1], A_ub=A_ub, b_ub=b_ub, A_eq=[[1] * n + [0]], b_eq=[1],
                bounds=[(0, 1)] * n + [(0, None)], method='highs')
    return round(r.fun, 6)


# ---------------- Chapter 2 (foundations)
ell = [([0], 1 / 3, 1 / 3), ([1, 2], 2 / 3, 2 / 3)]          # Ellsberg urn (R, B, Y)
R['ellsberg'] = {name: lp_bounds(3, ell, ev) for name, ev in
                 [('red', [0]), ('black', [1]), ('yellow', [2]), ('red_or_black', [0, 1]), ('black_or_yellow', [1, 2])]}
p = np.array([0.4, 0.7])                                        # coin, gamble 10 if H, -5 if T
R['coin_gamble'] = [float(min(15 * p - 5)), float(max(15 * p - 5))]
R['sureloss_ab'] = sure_loss(2, [([0], 0.6, 1), ([1], 0.5, 1)])
cons = [([0], 0.2, 1), ([1], 0.3, 1), ([0, 1], 0.6, 1)]         # natural extension example
R['natext'] = {str(ev): lp_bounds(3, cons, ev) for ev in [[0], [1], [2], [0, 1], [0, 2], [1, 2]]}
foc = {('a',): 0.3, ('b', 'c'): 0.4, ('a', 'b', 'c'): 0.3}       # belief function
Om = {'a', 'b', 'c'}
evs = [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c'), ('b', 'c')]
bel = lambda A: sum(w for f, w in foc.items() if set(f) <= set(A))
pl = lambda A: sum(w for f, w in foc.items() if set(f) & set(A))
R['belief'] = {''.join(A): dict(Bel=round(bel(A), 4), Pl=round(pl(A), 4),
               triple=[round(bel(A), 4), round(pl(A) - bel(A), 4), round(bel(tuple(Om - set(A))), 4)]) for A in evs}
R['straddle'] = {''.join(A): round(sum(w for f, w in foc.items() if set(f) & set(A) and not set(f) <= set(A)), 4)
                 for A in evs}
u, v = 0.6, 0.7
clay = lambda u, v, t=2.0: (u ** -t + v ** -t - 1) ** (-1 / t)
R['copulas_06_07'] = dict(Pi=round(u * v, 4), M=min(u, v), W=round(max(0, u + v - 1), 4), Clayton2=round(clay(u, v), 4),
                          Clayton2_survival=round(u + v - 1 + clay(1 - u, 1 - v), 4))
x = (0.6, 0.2, 0.3); y = (0.5, 0.4, 0.2)
R['nnorms'] = dict(algebraic=[round(x[0] * y[0], 4), round(x[1] + y[1] - x[1] * y[1], 4), round(x[2] + y[2] - x[2] * y[2], 4)],
                   minmax=[min(x[0], y[0]), max(x[1], y[1]), max(x[2], y[2])],
                   frechet=[round(max(0, x[0] + y[0] - 1), 4), max(x[1], y[1]), max(x[2], y[2])])
a, b = 0.7, 0.5
R['plith_op'] = {str(c): dict(conj=round((1 - c) * a * b + c * (a + b - a * b), 4),
                              disj=round((1 - c) * (a + b - a * b) + c * a * b, 4)) for c in [0, 0.25, 0.5, 0.75, 1]}

# ---------------- Chapter 4
c1 = [([0], 0.4, 0.7), ([1], 0.4, 0.7), ([2], 0.4, 0.7), ([1, 2], 0.3, 0.6), ([0, 2], 0.3, 0.6), ([0, 1], 0.3, 0.6)]
R['C1_delta'] = sure_loss(3, c1); R['C1_delta_exact'] = str(Fr(1, 15))
tab1 = {(0,): (0.5, 0.5, 0.0), (1,): (0.0, 0.5, 0.5), (2,): (0.0, 0.8, 0.2), (0, 1): (0.2, 0.8, 0.0),
        (0, 2): (0.5, 0.5, 0.0), (1, 2): (0.0, 0.5, 0.5)}
cons = [(list(e), t[0], 1 - t[2]) for e, t in tab1.items()]
corr = {}
for e in tab1:
    lo, hi = lp_bounds(3, cons, list(e)); corr[str(e)] = [round(lo, 4), round(hi - lo, 4), round(1 - hi, 4)]
R['C2_correction'] = corr

# ---------------- Chapter 5: singletons, four outcomes
T = np.array([0.1, 0.2, 0.3, 0.1]); I = np.array([0.2, 0.1, 0.3, 0.2]); D = 1 - T.sum()
R['sing4'] = dict(Delta=round(D, 4), sumI=round(I.sum(), 4), proper=bool(0 <= D <= I.sum() + 1e-12),
                  reach=[bool(I[i] <= D + 1e-12 and D <= I.sum() - I[i] + 1e-12) for i in range(4)])
cons = [([i], T[i], T[i] + I[i]) for i in range(4)]
ne = {}
for A in [(0, 1), (2,), (0, 2, 3), (1, 3)]:
    TA = T[list(A)].sum(); IA = I[list(A)].sum(); IAc = I.sum() - IA
    L = TA + max(D - IAc, 0); U = TA + min(IA, D)
    ne[str(A)] = dict(closed=[round(L, 4), round(U, 4)], lp=lp_bounds(4, cons, list(A)),
                      triple=[round(L, 4), round(U - L, 4), round(1 - U, 4)])
R['sing4']['natext'] = ne
I2 = np.array([0.4, 0.1, 0.3, 0.2]); cons2 = [([i], T[i], T[i] + I2[i]) for i in range(4)]
R['sing4_bad'] = dict(Delta=round(D, 4), reach=[bool(I2[i] <= D + 1e-12 and D <= I2.sum() - I2[i] + 1e-12) for i in range(4)],
                      corrected=[list(lp_bounds(4, cons2, [i])) for i in range(4)])
n = np.array([6, 3, 1]); N = n.sum(); s = 2
R['idm631'] = dict(T=[round(x, 4) for x in n / (N + s)], I=round(s / (N + s), 4),
                   L_01=round((n[0] + n[1]) / (N + s), 4), U_01=round((n[0] + n[1] + s) / (N + s), 4))

# ---------------- Chapter 6
R['thm3_ex'] = dict(triple=[0.7, 0.1, 0.6], delta=round(max(0.7 + 0.6 - 1, 0) / 2, 4))
R['sym_reading'] = dict(t_02_03_02=[round((1 + 0.0 - 0.3) / 2, 4), round((1 + 0.0 + 0.3) / 2, 4)])

# ---------------- Chapter 7: saturating chart r=6, q=3, W=2
r_, q_, W = 6, 3, 2; Ts = r_ / (r_ + W); Fs = q_ / (q_ + W)
b_ = Ts * (1 - Fs) / (1 - Ts * Fs); d_ = Fs * (1 - Ts) / (1 - Ts * Fs); u_ = (1 - Ts) * (1 - Fs) / (1 - Ts * Fs)
R['sl_chart'] = dict(Ts=Ts, Fs=Fs, sum=round(Ts + Fs, 4), rq=r_ * q_, W2=W * W, b=round(b_, 4), d=round(d_, 4), u=round(u_, 4),
                     b_direct=str(Fr(r_, r_ + q_ + W)), d_direct=str(Fr(q_, r_ + q_ + W)), u_direct=str(Fr(W, r_ + q_ + W)),
                     interval=[round(b_, 4), round(b_ + u_, 4)], sure_loss=round((Ts + Fs - 1) / 2, 4))

# ---------------- Chapter 8: glut lifting of (0.8, 0.3, 0.7); atoms (T,I,F) patterns
pats = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]


def glut_lp(x, obj):
    E = np.array([[p[j] for p in pats] for j in range(3)], float)
    r = linprog(np.array(obj, float), A_ub=-E, b_ub=[-v for v in x], A_eq=[np.ones(4)], b_eq=[1],
                bounds=[(0, 1)] * 4, method='highs')
    return round(r.fun, 6), [round(t, 6) for t in r.x]


xg = (0.8, 0.3, 0.7)
E = [[p[j] for p in pats] for j in range(3)]
R['glut_08_03_07'] = {k: glut_lp(xg, E[j]) for j, k in enumerate('TIF')}
R['glut_08_03_07']['min_mass_atom_101'] = glut_lp(xg, [0, 1, 0, 0])[0]
R['glut_08_03_07']['min_P_ET_and_EF'] = glut_lp(xg, [0, 1, 0, 1])[0]
x1 = (0.6, 0.2, 0.2); x2 = (0.5, 0.4, 0.1); u1, u2 = 1 - x1[2], 1 - x2[2]
R['thm7'] = dict(algebraic=[round(x1[0] * x2[0], 4), round(x1[1] + x2[1] - x1[1] * x2[1], 4), round(x1[2] + x2[2] - x1[2] * x2[2], 4)],
                 credal_indep=[round(x1[0] * x2[0], 4), round(u1 * u2, 4)], I_credal=round(u1 * u2 - x1[0] * x2[0], 4),
                 sigma=round(x1[1] * x2[2] + x2[1] * x1[2], 4), comon=[min(x1[0], x2[0]), min(u1, u2)],
                 minmax=[min(x1[0], x2[0]), max(x1[1], x2[1]), max(x1[2], x2[2])],
                 surplus_minmax=round(max(x1[1], x2[1]) - (min(u1, u2) - min(x1[0], x2[0])), 4),
                 frechet=[round(max(0, x1[0] + x2[0] - 1), 4), max(x1[1], x2[1]), max(x1[2], x2[2])])

# ---------------- Chapter 12: plithogenic N-norm, x1=(0.6,0.2,0.3), x2=(0.5,0.4,0.2)
x1 = (0.6, 0.2, 0.3); x2 = (0.5, 0.4, 0.2)
Pi = lambda a, b: a * b; Mm = lambda a, b: min(a, b); Ww = lambda a, b: max(0, a + b - 1)


def Npl(CT, CF, c):
    return [round(c * (x1[0] + x2[0]) + (1 - 2 * c) * CT(x1[0], x2[0]), 4), round((x1[1] + x2[1]) / 2, 4),
            round((1 - c) * (x1[2] + x2[2]) - (1 - 2 * c) * CF(x1[2], x2[2]), 4)]


R['plith_nnorm'] = {str(c): dict(indep=Npl(Pi, Pi, c), comon=Npl(Mm, Mm, c), countermon=Npl(Ww, Ww, c),
                                 no_assumption=(Npl(Ww, Mm, c) if c <= 0.5 else Npl(Mm, Ww, c)))
                    for c in [0, 0.3, 0.5, 0.8]}
f = lambda a, b, c=0.3: c * (a + b) + (1 - 2 * c) * a * b
a, b, d = 0.2, 0.5, 0.9
R['assoc'] = dict(left=round(f(f(a, b), d), 6), right=round(f(a, f(b, d)), 6), diff=round(f(f(a, b), d) - f(a, f(b, d)), 6),
                  formula=round(0.3 * 0.7 * (d - a), 6))

# ---------------- exercise answers
R['ex'] = dict(idm_843=dict(T=[round(t, 4) for t in np.array([8, 4, 3]) / 17], I=round(2 / 17, 4)),
               sl_r2q8=dict(Ts=round(2 / 4, 4), Fs=round(8 / 10, 4), rq=16, W2=4),
               delta_09_04=round((0.9 + 0.4 - 1) / 2, 4),
               sigma_pi_03_02_05__04_04_02=round(0.2 * 0.2 + 0.4 * 0.5, 4),
               mean_03_06_08=round((0.3 + 0.6 + 0.8) / 3, 4))
json.dump(R, open('foundations_examples.json', 'w'), indent=1)
print(json.dumps(R, indent=1))

# ---------------- extra checks for exercises (iteration 2)
T3 = [0.2, 0.2, 0.2]; I3 = [0.5, 0.1, 0.1]
cons3 = [([i], T3[i], T3[i] + I3[i]) for i in range(3)]
R['ex_ch5_2'] = dict(Delta=round(1 - sum(T3), 4), attained=[list(lp_bounds(3, cons3, [i])) for i in range(3)])
R['ex_ch5_1'] = {str(A): lp_bounds(4, [([i], T[i], T[i] + I[i]) for i in range(4)], list(A)) for A in [(2,), (1, 3)]}
json.dump(R, open('foundations_examples.json', 'w'), indent=1)
print('ex_ch5_2', R['ex_ch5_2'], 'ex_ch5_1', R['ex_ch5_1'])

# ---------------- iteration 3: applications of Chapters 12-13
import itertools as _it
f3 = lambda a, b, c: c * (a + b) + (1 - 2 * c) * a * b          # binary plithogenic conjunction, product copula
xs = [0.3, 0.6, 0.8, 0.5]; cc = 0.3
seq = {}
for perm in _it.permutations(range(4)):
    v = xs[perm[0]]
    for k in perm[1:]:
        v = f3(v, xs[k], cc)
    seq[perm] = v
vals = sorted(seq.values())
prodx = float(np.prod(xs)); un = 1 - float(np.prod([1 - x for x in xs]))
R['group4'] = dict(inputs=xs, c=cc, seq_min=round(vals[0], 4), seq_max=round(vals[-1], 4), n_orders=len(seq),
                   n_distinct=len(set(round(v, 10) for v in vals)),
                   nary_indep=round((1 - cc) * prodx + cc * un, 4), nary_comon=round((1 - cc) * min(xs) + cc * max(xs), 4),
                   mean=round(sum(xs) / 4, 4))
# medical example of Section 11.1: x1=(0.70,0.20,0.10), x2=(0.55,0.25,0.20); truth with no dependence assumption
T1, T2 = 0.70, 0.55
Tno = lambda c: c * (T1 + T2) + (1 - 2 * c) * max(0, T1 + T2 - 1) if c <= 0.5 else c * (T1 + T2) + (1 - 2 * c) * min(T1, T2)
cs = np.linspace(0, 1, 100001)
first = next(c for c in cs if Tno(c) >= 0.5 - 1e-12)
R['medical_threshold'] = dict(T_noassump_c0=round(Tno(0), 4), T_noassump_c025=round(Tno(0.25), 4), c_star=round(float(first), 4),
                              exact='1/3', T_at_1_3=round(Tno(1 / 3), 6))
json.dump(R, open('foundations_examples.json', 'w'), indent=1)
print('group4', R['group4']); print('medical', R['medical_threshold'])
