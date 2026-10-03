"""Numbers of the section "Choosing an N-norm in practice: dependence, copulas and robustness".

Every number quoted in the section comes from this script. Output: nnorm_choice.json,
fig_theta_scores.png, nnorm_choice_log.txt.
Run: python nnorm_choice.py   (numpy, scipy, matplotlib; about one minute)

Parts
 A  copulas, survival copulas, surplus sigma_C on normalised triples
 B  radial symmetry: face identity sigma = C_hat - C; Clayton/Gumbel negative surplus at explicit points;
    shuffle S of Proposition 2 of the companion paper (exact rationals); grid minima for Pi, M, W, Frank
 C  Kendall's tau <-> parameter: closed forms vs numerical integration; simulation and recovery of theta
 D  tail dependence coefficients (closed form vs C(t,t)/t, (1-2t+C(t,t))/(1-t))
 E  MCDM example: Frank N-norm AND of three criteria, theta sweep, W-M bracket, crossing points
 F  estimating theta from synthetic evidence counts of two sources (Subjective Logic triples), bootstrap
"""
import json, math, os
from fractions import Fraction as Fr
import numpy as np
from scipy import integrate, optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.getcwd()  # book edition: outputs go to the current working directory
os.makedirs(RES, exist_ok=True)
OUT = {}
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.append(s)


# ---------------------------------------------------------------- A. copulas
def Pi(u, v): return u * v
def M(u, v): return min(u, v)
def W(u, v): return max(0.0, u + v - 1.0)


def frank(th):
    if th == 0:
        return Pi
    def Cpos(u, v, t):
        # stable form for t > 0: C = m - (1/t) log[(1 + e^{-t(Mx-m)} - e^{-t Mx} - e^{-t(1-m)}) / (1 - e^{-t})]
        if u <= 0 or v <= 0: return 0.0
        m, Mx = min(u, v), max(u, v)
        arg = (1 + math.exp(-t * (Mx - m)) - math.exp(-t * Mx) - math.exp(-t * (1 - m))) / (-math.expm1(-t))
        return min(max(m - math.log(arg) / t, max(0.0, u + v - 1)), m)
    def C(u, v):
        if th > 0: return Cpos(u, v, th)
        # Frank reflection: C_{-t}(u, v) = u - C_t(u, 1 - v)
        return min(max(u - Cpos(u, 1 - v, -th), max(0.0, u + v - 1)), min(u, v))
    return C


def clayton(th):
    def C(u, v):
        if u <= 0 or v <= 0: return 0.0
        return max(u ** -th + v ** -th - 1.0, 0.0) ** (-1.0 / th)
    return C


def gumbel(th):
    def C(u, v):
        if u <= 0 or v <= 0: return 0.0
        return math.exp(-((-math.log(u)) ** th + (-math.log(v)) ** th) ** (1.0 / th))
    return C


def survival(C):
    return lambda u, v: u + v - 1.0 + C(1.0 - u, 1.0 - v)


def sigma(C, x1, x2):
    (T1, I1, F1), (T2, I2, F2) = x1, x2
    return I1 + I2 - C(I1, I2) - C(1 - F1, 1 - F2) + C(T1, T2)


def simplex(n):
    return [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]


def frank_textbook(th):
    return lambda u, v: -math.log1p(math.expm1(-th * u) * math.expm1(-th * v) / math.expm1(-th)) / th
OUT["frank_stable_vs_textbook"] = max(abs(frank(t)(u, v) - frank_textbook(t)(u, v)) for t in (-10, -3, -0.5, 0.5, 3, 10)
                                      for u in [i / 17 for i in range(1, 17)] for v in [j / 13 for j in range(1, 13)])
log("Frank stable form vs textbook formula, max difference", OUT["frank_stable_vs_textbook"])

# ---------------------------------------------------------------- B. radial symmetry
log("== B. radial symmetry and the surplus ==")
# face identity on F1=F2=0: sigma = C_hat(I1,I2) - C(I1,I2)  (checked for several copulas, incl. non-symmetric)
g = [k / 20 for k in range(21)]
face_err = 0.0
for C in (Pi, M, W, frank(5.0), clayton(2.0), gumbel(2.0)):
    Ch = survival(C)
    for a in g:
        for b in g:
            face_err = max(face_err, abs(sigma(C, (1 - a, a, 0.0), (1 - b, b, 0.0)) - (Ch(a, b) - C(a, b))))
log("face identity max error", face_err)
OUT["face_identity_max_error"] = face_err

# Clayton / Gumbel: negative surplus at explicit points (theta = 2)
neg = {}
for name, C in (("Clayton(2)", clayton(2.0)), ("Gumbel(2)", gumbel(2.0))):
    best = None
    for a in [k / 100 for k in range(101)]:
        for b in [k / 100 for k in range(101)]:
            s = sigma(C, (1 - a, a, 0.0), (1 - b, b, 0.0))
            if best is None or s < best[0]:
                best = (s, a, b)
    neg[name] = dict(min_on_face=best[0], argmin_I1=best[1], argmin_I2=best[2])
    log(name, "min sigma on face F=0 (step 0.01):", round(best[0], 6), "at I1, I2 =", best[1], best[2])
# a clean, easily recomputed point for the text: x1 = x2 = (0.7, 0.3, 0)
xc = (0.7, 0.3, 0.0)
C2 = clayton(2.0); Ch2 = survival(C2)
clean = dict(x=xc, C_I=C2(0.3, 0.3), C_one=C2(1.0, 1.0), C_T=C2(0.7, 0.7), sigma=sigma(C2, xc, xc),
             Chat_03=Ch2(0.3, 0.3))
log("Clayton(2) at x1=x2=(0.7,0.3,0): C(0.3,0.3)=%.6f C(0.7,0.7)=%.6f sigma=%.6f  [= C_hat(0.3,0.3)-C(0.3,0.3) = %.6f-%.6f]"
    % (clean["C_I"], clean["C_T"], clean["sigma"], clean["Chat_03"], clean["C_I"]))
G2 = gumbel(2.0)
cleanG = dict(x=(0.3, 0.7, 0.0), sigma=sigma(G2, (0.3, 0.7, 0.0), (0.3, 0.7, 0.0)), C_I=G2(0.7, 0.7), C_T=G2(0.3, 0.3))
log("Gumbel(2) at x1=x2=(0.3,0.7,0): C(0.7,0.7)=%.6f C(0.3,0.3)=%.6f sigma=%.6f" % (cleanG["C_I"], cleanG["C_T"], cleanG["sigma"]))
OUT["non_symmetric_negative"] = dict(face_minima=neg, clayton2_clean=clean, gumbel2_clean=cleanG)

# shuffle S (Proposition 2 of the companion paper) in exact rationals
segs = [(Fr(0), Fr(0)), (Fr(1, 4), Fr(1, 4)), (Fr(1, 2), Fr(-1, 4)), (Fr(3, 4), Fr(0))]
def S(u, v):
    tot = Fr(0)
    for a, s in segs:
        hi = min(a + Fr(1, 4), u, v - s)
        if hi > a: tot += hi - a
    return tot
N = 24; gq = [Fr(i, N) for i in range(N + 1)]
S_ok = (all(S(u, Fr(0)) == 0 and S(Fr(0), u) == 0 and S(u, Fr(1)) == u and S(Fr(1), u) == u for u in gq)
        and all(S(gq[i + 1], gq[j + 1]) - S(gq[i], gq[j + 1]) - S(gq[i + 1], gq[j]) + S(gq[i], gq[j]) >= 0 for i in range(N) for j in range(N))
        and all(S(u, v) == u + v - 1 + S(1 - u, 1 - v) for u in gq for v in gq)
        and all(S(u, v) == S(v, u) for u in gq for v in gq))
xs = (Fr(1, 2), Fr(1, 4), Fr(1, 4))
sS = xs[1] + xs[1] - S(xs[1], xs[1]) - S(1 - xs[2], 1 - xs[2]) + S(xs[0], xs[0])
log("shuffle S: copula + radially symmetric + exchangeable on grid 1/24:", S_ok, " sigma_S(x,x) =", sS)
OUT["shuffle"] = dict(checks_grid_1_24=S_ok, sigma=str(sS))

# grid minima of sigma over pairs of the simplex grid of step 1/20 (231 points, 53 361 pairs)
P = simplex(20)
grid_min = {}
for name, C in [("Pi", Pi), ("M", M), ("W", W)] + [("Frank(%g)" % t, frank(t)) for t in (-20, -5, -1, 1, 5, 20)] \
        + [("Clayton(2)", clayton(2.0)), ("Gumbel(2)", gumbel(2.0))]:
    m = min(sigma(C, a, b) for a in P for b in P)
    grid_min[name] = m
    log("min sigma on simplex grid 1/20:", name, "%.3e" % m)
OUT["grid_minima_step_1_20"] = grid_min
# W surplus closed form check: sigma_W = a - (a-1)^+ - (t+a)^+ + t^+, a = I1+I2, t = T1+T2-1
errW = max(abs(sigma(W, x, y) - ((x[1] + y[1]) - max(x[1] + y[1] - 1, 0) - max(x[0] + y[0] - 1 + x[1] + y[1], 0) + max(x[0] + y[0] - 1, 0)))
           for x in P for y in P)
log("sigma_W closed form max error", errW)
OUT["sigma_W_closed_form_error"] = errW

# ---------------------------------------------------------------- C. Kendall's tau <-> theta
log("== C. Kendall tau and copula parameters ==")
def debye1(x):
    if x == 0: return 1.0
    val = integrate.quad(lambda t: t / math.expm1(t) if t != 0 else 1.0, 0, abs(x))[0] / abs(x)
    return val if x > 0 else val + abs(x) / 2  # D1(-x) = D1(x) + x/2
def tau_frank(th): return 0.0 if th == 0 else 1 - 4 / th * (1 - debye1(th))
def tau_clayton(th): return th / (th + 2)
def tau_gumbel(th): return 1 - 1 / th
def theta_clayton(tau): return 2 * tau / (1 - tau)
def theta_gumbel(tau): return 1 / (1 - tau)
def theta_frank(tau):
    if abs(tau) < 1e-12: return 0.0
    return optimize.brentq(lambda t: tau_frank(t) - tau, -200, 200) if abs(tau) < 0.97 else float("nan")


def tau_numeric(C, h=1e-6):
    """tau = 1 - 4 int int dC/du dC/dv du dv (Nelsen 2006, Theorem 5.1.1 form), by quadrature."""
    def du(u, v): return (C(min(u + h, 1), v) - C(max(u - h, 0), v)) / (min(u + h, 1) - max(u - h, 0))
    def dv(u, v): return (C(u, min(v + h, 1)) - C(u, max(v - h, 0))) / (min(v + h, 1) - max(v - h, 0))
    n = 400; grid = (np.arange(n) + 0.5) / n
    tot = sum(du(u, v) * dv(u, v) for u in grid for v in grid) / n ** 2
    return 1 - 4 * tot

closed_vs_numeric = {}
for name, C, tf, th in (("Clayton", clayton, tau_clayton, 2.0), ("Gumbel", gumbel, tau_gumbel, 2.0),
                        ("Frank", frank, tau_frank, 5.0), ("Frank", frank, tau_frank, -5.0)):
    tn = tau_numeric(C(th)); tc = tf(th)
    closed_vs_numeric["%s(%g)" % (name, th)] = dict(closed=tc, numeric=tn)
    log("%s(%g): tau closed form %.4f, numerical integration %.4f" % (name, th, tc, tn))
OUT["tau_closed_vs_numeric"] = closed_vs_numeric
# inversion round trip
rt = max(abs(theta_frank(tau_frank(t)) - t) for t in (-20, -5, -1, -0.2, 0.2, 1, 5, 20))
log("Frank inversion round-trip max error", rt)
OUT["frank_inversion_roundtrip_error"] = rt
OUT["frank_theta_at_tau_0_5"] = theta_frank(0.5)
log("Frank theta for tau = 0.5: %.3f" % OUT["frank_theta_at_tau_0_5"])
OUT["frank_tau_table"] = {str(t): tau_frank(t) for t in (-20, -10, -5, -2, -1, 1, 2, 5, 10, 20)}


# samplers (conditional inversion for Clayton and Frank, Marshall-Olkin with positive stable frailty for Gumbel)
def sample_clayton(th, n, rng):
    u = rng.random(n); w = rng.random(n)
    v = ((w ** (-th / (1 + th)) - 1) * u ** (-th) + 1) ** (-1 / th)
    return u, v
def sample_frank(th, n, rng):
    u = rng.random(n); w = rng.random(n)
    v = -np.log1p(w * np.expm1(-th) / (w + (1 - w) * np.exp(-th * u))) / th
    return u, v
def sample_gumbel(th, n, rng):
    a = 1 / th
    U = rng.uniform(0, math.pi, n); E = rng.exponential(1, n)
    Vs = (np.sin(a * U) / np.sin(U) ** (1 / a)) * (np.sin((1 - a) * U) / E) ** ((1 - a) / a)  # Kanter: LT exp(-s^a)
    E1 = rng.exponential(1, n); E2 = rng.exponential(1, n)
    return np.exp(-(E1 / Vs) ** a), np.exp(-(E2 / Vs) ** a)

rng = np.random.default_rng(20261001)
sim = {}
for name, smp, inv, th in (("Clayton", sample_clayton, theta_clayton, 2.0), ("Gumbel", sample_gumbel, theta_gumbel, 2.0),
                           ("Frank", sample_frank, theta_frank, 5.0), ("Frank", sample_frank, theta_frank, -5.0)):
    reps = []
    for r in range(20):
        u, v = smp(th, 2000, rng)
        tau = stats.kendalltau(u, v).statistic
        reps.append((tau, inv(tau)))
    taus = np.array([x[0] for x in reps]); ths = np.array([x[1] for x in reps])
    sim["%s(%g)" % (name, th)] = dict(n=2000, reps=20, mean_tau=float(taus.mean()), mean_theta_hat=float(ths.mean()),
                                      sd_theta_hat=float(ths.std(ddof=1)), true_tau=float({"Clayton": tau_clayton, "Gumbel": tau_gumbel, "Frank": tau_frank}[name](th)))
    log("%s(%g): true tau %.4f; 20 samples of n=2000: mean tau %.4f, mean theta_hat %.3f (sd %.3f)"
        % (name, th, sim["%s(%g)" % (name, th)]["true_tau"], taus.mean(), ths.mean(), ths.std(ddof=1)))
OUT["simulation_recovery"] = sim

# ---------------------------------------------------------------- D. tail dependence
log("== D. tail dependence ==")
tails = {}
for name, C, lamL, lamU in (("Clayton(2)", clayton(2.0), 2 ** -0.5, 0.0), ("Gumbel(2)", gumbel(2.0), 0.0, 2 - 2 ** 0.5),
                            ("Frank(5)", frank(5.0), 0.0, 0.0)):
    t = 1e-6
    numL = C(t, t) / t; numU = (1 - 2 * (1 - t) + C(1 - t, 1 - t)) / t
    tails[name] = dict(lambda_L_closed=lamL, lambda_L_numeric=numL, lambda_U_closed=lamU, lambda_U_numeric=numU)
    log("%s: lambda_L %.4f (numeric %.4f), lambda_U %.4f (numeric %.4f)" % (name, lamL, numL, lamU, numU))
OUT["tail_dependence"] = tails


# ---------------------------------------------------------------- E. MCDM example
log("== E. MCDM example ==")
ALTS = {  # three criteria per alternative: (T, I, F), normalised
    "A1": [(0.70, 0.20, 0.10), (0.70, 0.15, 0.15), (0.70, 0.20, 0.10)],   # balanced
    "A2": [(0.95, 0.05, 0.00), (0.95, 0.00, 0.05), (0.45, 0.25, 0.30)],   # two excellent, one weak
    "A3": [(0.80, 0.10, 0.10), (0.85, 0.10, 0.05), (0.60, 0.25, 0.15)],   # good, one moderate
    "A4": [(0.90, 0.05, 0.05), (0.60, 0.30, 0.10), (0.72, 0.13, 0.15)],   # mixed
}


def tnorm_n(C, xs):
    out = xs[0]
    for x in xs[1:]: out = C(out, x)
    return out
def tconorm_n(C, xs):
    out = xs[0]
    for x in xs[1:]: out = out + x - C(out, x)
    return out
def nnorm(C, triples):
    return (tnorm_n(C, [t[0] for t in triples]), tconorm_n(C, [t[1] for t in triples]), tconorm_n(C, [t[2] for t in triples]))
def score(x):
    T, I, F = x
    return (2 + T - I - F) / 3


def agg(Cname):
    C = {"W": W, "Pi": Pi, "M": M}[Cname] if isinstance(Cname, str) else frank(Cname)
    return {a: nnorm(C, ALTS[a]) for a in ALTS}

table = {}
for key in ("W", -20.0, -5.0, -1.0, "Pi", 1.0, 5.0, 20.0, "M"):
    res = agg(key)
    sc = {a: score(res[a]) for a in res}
    rank = sorted(sc, key=lambda a: -sc[a])
    table[str(key)] = dict(triples={a: [round(v, 4) for v in res[a]] for a in res}, scores=sc, ranking=rank)
    log("%-6s" % key, " ".join("%s=%.4f" % (a, sc[a]) for a in ALTS), " ranking:", " > ".join(rank))
OUT["mcdm_table"] = table
# order independence of the n-ary Frank N-norm (associativity) check
perm_err = 0.0
for th in (-20.0, -5.0, 5.0, 20.0):
    C = frank(th)
    for a in ALTS:
        x = ALTS[a]
        for p in ((0, 1, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0)):
            y = [x[i] for i in p]
            perm_err = max(perm_err, max(abs(u - v) for u, v in zip(nnorm(C, x), nnorm(C, y))))
log("order independence (associativity) max error", perm_err)
OUT["order_independence_error"] = perm_err

# theta sweep, monotonicity, crossings
thetas = np.concatenate([-np.logspace(2, -2, 200), [0.0], np.logspace(-2, 2, 200)])
curves = {a: [] for a in ALTS}
for th in thetas:
    C = frank(float(th))
    for a in ALTS:
        curves[a].append(score(nnorm(C, ALTS[a])))
mono = all(all(np.diff(curves[a]) >= -1e-12) for a in ALTS)
log("scores non-decreasing in theta on the sweep:", mono)
OUT["scores_monotone_in_theta"] = bool(mono)
crossings = []
names = list(ALTS)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        d = np.array(curves[a]) - np.array(curves[b])
        for k in range(len(thetas) - 1):
            if d[k] * d[k + 1] < 0 and min(abs(d[k]), abs(d[k + 1])) > 0 or (abs(d[k]) > 1e-9 and abs(d[k + 1]) > 1e-9 and d[k] * d[k + 1] < 0):
                f = lambda t: score(nnorm(frank(t), ALTS[a])) - score(nnorm(frank(t), ALTS[b]))
                r = optimize.brentq(f, thetas[k], thetas[k + 1])
                crossings.append(dict(pair=a + "-" + b, theta=float(r), tau=float(tau_frank(r))))
crossings.sort(key=lambda c: c["theta"])
for c in crossings:
    log("crossing %s at theta = %.3f (Kendall tau %.3f)" % (c["pair"], c["theta"], c["tau"]))
OUT["crossings"] = crossings
# ranking regimes between crossings
cuts = [-1e9] + [c["theta"] for c in crossings] + [1e9]
regimes = []
for lo, hi in zip(cuts[:-1], cuts[1:]):
    mid = (max(lo, -100) + min(hi, 100)) / 2 if not (lo == -1e9 and hi == 1e9) else 0
    if lo == -1e9: mid = hi - 1 if hi < 1e9 else 0
    if hi == 1e9: mid = lo + 1
    sc = {a: score(nnorm(frank(mid), ALTS[a])) for a in ALTS}
    regimes.append(dict(theta_from=lo if lo > -1e9 else "-inf (W)", theta_to=hi if hi < 1e9 else "+inf (M)",
                        ranking=sorted(sc, key=lambda a: -sc[a])))
for r in regimes: log("regime", r)
OUT["regimes"] = regimes
# W-M bracket per alternative and robustness of pairwise comparisons
brk = {a: (table["W"]["scores"][a], table["M"]["scores"][a]) for a in ALTS}
dominance = {}
for a in ALTS:
    for b in ALTS:
        if a != b and brk[a][0] > brk[b][1]:
            dominance.setdefault(a, []).append(b)
log("W-M score brackets", {a: tuple(round(v, 4) for v in brk[a]) for a in brk})
log("interval dominance (lower score of a > upper score of b):", dominance)
OUT["bracket"] = brk; OUT["interval_dominance"] = dominance
# bracket property: every Frank score lies in [score_W, score_M]
inside = all(brk[a][0] - 1e-12 <= s <= brk[a][1] + 1e-12 for a in ALTS for s in curves[a])
log("all Frank scores inside the W-M bracket:", inside)
OUT["frank_inside_bracket"] = bool(inside)
# two-criteria check of Theorem 9(b) triple vs bracket for A2 (first two criteria)
x, y = ALTS["A2"][0], ALTS["A2"][2]
t9b = (max(0, x[0] + y[0] - 1), max(x[1], y[1]), max(x[2], y[2]))
OUT["theorem9b_example"] = dict(x=x, y=y, triple=t9b, NW=nnorm(W, [x, y]), NM=nnorm(M, [x, y]))
log("Theorem 9(b) triple for A2 criteria 1 and 3:", t9b, " N_W:", nnorm(W, [x, y]), " N_M:", nnorm(M, [x, y]))

# figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200)
styles = {"A1": ("-", "#1b7837"), "A2": ("--", "#762a83"), "A3": ("-.", "#e08214"), "A4": (":", "#2166ac")}
xplot = np.sign(thetas) * np.log10(1 + np.abs(thetas))
for a in ALTS:
    ls, col = styles[a]
    ax.plot(xplot, curves[a], ls, color=col, lw=1.8, label=a)
    ax.plot([-2.25], [brk[a][0]], marker="<", color=col, ms=6)
    ax.plot([2.25], [brk[a][1]], marker=">", color=col, ms=6)
for c in crossings:
    xc_ = np.sign(c["theta"]) * np.log10(1 + abs(c["theta"]))
    ax.axvline(xc_, color="0.6", lw=0.7)
ticks = [-100, -20, -5, -1, 0, 1, 5, 20, 100]
ax.set_xticks([np.sign(t) * np.log10(1 + abs(t)) for t in ticks]); ax.set_xticklabels([str(t) for t in ticks])
ax.set_xlim(-2.4, 2.4)
ax.text(-2.25, 0.805, "W", ha="center", fontsize=9); ax.text(2.25, 0.805, "M", ha="center", fontsize=9)
ax.set_xlabel("Frank parameter θ  (← W, countermonotone · 0 = Π · comonotone, M →)")
ax.set_ylabel("score s = (2 + T − I − F)/3")
ax.legend(loc="lower right", frameon=False, ncol=4)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_theta_scores.png"))
log("figure written")

# ---------------------------------------------------------------- F. theta from evidence counts of two sources
log("== F. theta from synthetic evidence counts ==")
rng = np.random.default_rng(7)
n_items, Wprior = 60, 2.0
p = rng.beta(2, 2, n_items)                       # latent support of each item
noise1 = rng.normal(0, 0.12, n_items); noise2 = rng.normal(0, 0.12, n_items)
p1 = np.clip(p + noise1, 0.01, 0.99); p2 = np.clip(p + noise2, 0.01, 0.99)
m1 = rng.integers(6, 15, n_items); m2 = rng.integers(6, 15, n_items)   # number of observations per source
r1 = rng.binomial(m1, p1); r2 = rng.binomial(m2, p2)
q1 = m1 - r1; q2 = m2 - r2
T1 = r1 / (m1 + Wprior); T2 = r2 / (m2 + Wprior)
tau = stats.kendalltau(T1, T2).statistic
th_F = theta_frank(tau); th_C = theta_clayton(tau); th_G = theta_gumbel(tau)
boot = []
for b in range(1000):
    idx = rng.integers(0, n_items, n_items)
    tb = stats.kendalltau(T1[idx], T2[idx]).statistic
    boot.append(theta_frank(tb))
lo, hi = np.percentile(boot, [5, 95])
log("first five items (r1,q1,r2,q2):", [(int(r1[i]), int(q1[i]), int(r2[i]), int(q2[i])) for i in range(5)])
log("Kendall tau-b between T1 and T2 over %d items: %.4f" % (n_items, tau))
log("theta: Frank %.3f (bootstrap 90%% interval [%.3f, %.3f]); Clayton %.3f; Gumbel %.3f" % (th_F, lo, hi, th_C, th_G))
# conjunction of the two sources for item 0 under the estimated Frank and at the interval ends
i0 = 0
x1 = (r1[i0] / (m1[i0] + Wprior), Wprior / (m1[i0] + Wprior), q1[i0] / (m1[i0] + Wprior))
x2 = (r2[i0] / (m2[i0] + Wprior), Wprior / (m2[i0] + Wprior), q2[i0] / (m2[i0] + Wprior))
conj = {lab: nnorm(frank(float(t)), [x1, x2]) for lab, t in (("theta_lo", lo), ("theta_hat", th_F), ("theta_hi", hi))}
conj["W"] = nnorm(W, [x1, x2]); conj["Pi"] = nnorm(Pi, [x1, x2]); conj["M"] = nnorm(M, [x1, x2])
for k, v in conj.items(): log("item 0 conjunction", k, tuple(round(z, 4) for z in v))
OUT["counts_example"] = dict(n_items=n_items, W_prior=Wprior, first5=[[int(r1[i]), int(q1[i]), int(r2[i]), int(q2[i])] for i in range(5)],
                             tau=tau, theta_frank=th_F, theta_frank_boot90=[lo, hi], theta_clayton=th_C, theta_gumbel=th_G,
                             item0=dict(x1=x1, x2=x2, conj={k: list(v) for k, v in conj.items()}))

# ---------------------------------------------------------------- G. exercise checks
log("== G. exercises ==")
x, y = (0.6, 0.2, 0.2), (0.5, 0.3, 0.2)
ex1 = {k: nnorm(C, [x, y]) for k, C in (("W", W), ("Pi", Pi), ("M", M))}
for k, v in ex1.items(): log("ex1", k, tuple(round(z, 4) for z in v), "score %.4f" % score(v))
ex1["sigma"] = {k: sigma(C, x, y) for k, C in (("W", W), ("Pi", Pi), ("M", M))}
log("ex1 sigma", ex1["sigma"])
ex2 = dict(tau=0.4, clayton=theta_clayton(0.4), gumbel=theta_gumbel(0.4), frank=theta_frank(0.4))
log("ex2 tau=0.4:", ex2)
C2 = clayton(2.0)
ex3 = dict(C_02=C2(0.2, 0.2), C_08=C2(0.8, 0.8), sigma=sigma(C2, (0.8, 0.2, 0.0), (0.8, 0.2, 0.0)))
log("ex3 Clayton(2) at x1=x2=(0.8,0.2,0):", ex3)
G2 = gumbel(2.0)
ex3g = dict(C_07=G2(0.7, 0.7), C_03=G2(0.3, 0.3), diag_check=abs(G2(0.7, 0.7) - 0.7 ** (2 ** 0.5)), sigma=sigma(G2, (0.3, 0.7, 0.0), (0.3, 0.7, 0.0)))
log("ex3 Gumbel(2) at x1=x2=(0.3,0.7,0):", ex3g)
OUT["exercises"] = dict(ex1={k: (list(v) if isinstance(v, tuple) else v) for k, v in ex1.items()}, ex2=ex2, ex3_clayton=ex3, ex3_gumbel=ex3g)

json.dump(OUT, open(os.path.join(RES, "nnorm_choice.json"), "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)
open(os.path.join(RES, "nnorm_choice_log.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
log("done")
