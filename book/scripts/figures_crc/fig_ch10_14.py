"""Figures of Chapters 10-14 of the proposed plan.

Fig_9_2  Choosing an N-norm (Section 8.3.4, Figure 8.3.1 of v12) redrawn in the common style: score of the
          three-criteria AND under the Frank N-norm along theta; crossings and W-M bracket checked against
          book_v11/chapters/ch_choosing_nnorm_files/results/nnorm_choice.json.
Fig_9_1  The surplus sigma_C on the face F1 = F2 = 0, where it equals C-hat - C (proof of Proposition 8.3.2), for
          Clayton(2) and Gumbel(2); minima checked against nnorm_choice.json.
Fig_11_1  Off values on the classical frame with a retraction budget (Proposition 8.4.1): where the off-credal set is
          non-empty and where it is faithful, for epsilon = 0.3; closed form checked by a linear programme over charges.
Fig_10_1  The contradiction degree as a dependence parameter (Theorem 10): truth of the plithogenic conjunction for
          the inputs of Exercise 13.1 (Section 11.1 data style) under three dependence models; markers: LP on 16 atoms.
Fig_14_1  The running example (Section 11.6): graduation interval under four dependence models, and what happens when
          one course is raised by 0.1; checked against book_v11/code/running_example/re05_decision_output.txt.
Fig_7_1  What event bounds can see (Proposition 3): the segment between two expert distributions and the larger core
          of its lower probability, with the data of Exercise 14.1(b); expected payoffs by linear programming.
"""
from common import *  # noqa: F401,F403
import itertools
import math
import re


# ---------------------------------------------------------------- Fig 10.1
ALTS = {  # Table 8.3.2 (book_v11/chapters/ch_choosing_nnorm_files/scripts/nnorm_choice.py)
    "A1": [(0.70, 0.20, 0.10), (0.70, 0.15, 0.15), (0.70, 0.20, 0.10)],
    "A2": [(0.95, 0.05, 0.00), (0.95, 0.00, 0.05), (0.45, 0.25, 0.30)],
    "A3": [(0.80, 0.10, 0.10), (0.85, 0.10, 0.05), (0.60, 0.25, 0.15)],
    "A4": [(0.90, 0.05, 0.05), (0.60, 0.30, 0.10), (0.72, 0.13, 0.15)],
}


def nnorm(C, xs):
    T, I, F = xs[0]
    for (t, i, f) in xs[1:]:
        T, I, F = C(T, t), I + i - C(I, i), F + f - C(F, f)
    return T, I, F


def score(x):
    return (2 + x[0] - x[1] - x[2]) / 3


def fig_10_1():
    J = json.load(open(book_result("nnorm_choice.json", "expected/added_sections/nnorm_choice.json"), encoding="utf-8"))
    thetas = np.concatenate([-np.logspace(2, -2, 200), [0.0], np.logspace(-2, 2, 200)])
    curves = {a: np.array([score(nnorm(frank(float(t)), ALTS[a])) for t in thetas]) for a in ALTS}
    brk = {a: (score(nnorm(Wc, ALTS[a])), score(nnorm(Mc, ALTS[a]))) for a in ALTS}
    for a in ALTS:
        assert abs(brk[a][0] - J["bracket"][a][0]) < 1e-9 and abs(brk[a][1] - J["bracket"][a][1]) < 1e-9
    from scipy.optimize import brentq
    cr = []
    for c in J["crossings"]:
        a, b = c["pair"].split("-")
        th = brentq(lambda t: score(nnorm(frank(t), ALTS[a])) - score(nnorm(frank(t), ALTS[b])), c["theta"] * 0.9, c["theta"] * 1.1)
        assert abs(th - c["theta"]) < 1e-6
        cr.append(dict(pair=c["pair"], theta=th, tau=c["tau"]))
    X = lambda t: np.sign(t) * np.log10(1 + np.abs(t))
    fig, ax = plt.subplots(figsize=(FULL, 2.9))
    for k, a in enumerate(ALTS):
        ax.plot(X(thetas), curves[a], ls=LINESTYLES[k], color=SERIES[k], lw=1.3, label=a)
        ax.plot([-2.27], [brk[a][0]], marker="<", color=SERIES[k], ms=5, mfc="white", mew=1.0)
        ax.plot([2.27], [brk[a][1]], marker=">", color=SERIES[k], ms=5, mfc="white", mew=1.0)
    for c in cr:
        ax.axvline(X(c["theta"]), color=MUTED, lw=0.5)
        ax.text(X(c["theta"]), 0.795, "%.3f" % c["tau"], rotation=90, fontsize=6, ha="right", va="top", color=INK2)
    ticks = [-100, -20, -5, -1, 0, 1, 5, 20, 100]
    ax.set_xticks([X(t) for t in ticks]); ax.set_xticklabels([str(t).replace("-", "−") for t in ticks])
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(0.36, 0.80)
    ax.text(-2.27, 0.368, "$W$", ha="center", fontsize=8); ax.text(2.27, 0.368, "$M$", ha="center", fontsize=8)
    ax.text(0.0, 0.368, "$\\Pi$", ha="center", fontsize=8)
    ax.set_xlabel("Frank parameter $\\theta$ (signed logarithmic scale); Kendall's $\\tau$ of each crossing above")
    ax.set_ylabel("score $s=(2+T-I-F)/3$")
    grid(ax)
    ax.legend(loc="upper left", ncol=2, fontsize=7.5, handlelength=2.4)
    save(fig, "Fig_9_2", dict(brackets={a: r(list(v)) for a, v in brk.items()}, crossings=[dict(pair=c["pair"], theta=r(c["theta"]), tau=r(c["tau"])) for c in cr],
                                curves_sample={a: r(list(curves[a][::40])) for a in ALTS}, theta_sample=r(list(thetas[::40]))))


def gumbel(th):
    def C(u, v):
        if u <= 0 or v <= 0:
            return 0.0
        return math.exp(-((-math.log(u)) ** th + (-math.log(v)) ** th) ** (1.0 / th))
    return C


def fig_10_2():
    J = json.load(open(book_result("nnorm_choice.json", "expected/added_sections/nnorm_choice.json"), encoding="utf-8"))
    g = np.round(np.linspace(0, 1, 101), 10)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.55), sharey=True)
    rec = {}
    levels = [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0]
    for ax, (name, C) in zip(axes, [("Clayton(2)", clayton(2.0)), ("Gumbel(2)", gumbel(2.0))]):
        Ch = survival(C)
        Z = np.array([[Ch(u, v) - C(u, v) for u in g] for v in g])
        k = np.unravel_index(np.argmin(Z), Z.shape)
        zmin, umin, vmin = Z[k], g[k[1]], g[k[0]]
        want = J["non_symmetric_negative"]["face_minima"][name]
        assert abs(zmin - want["min_on_face"]) < 1e-9 and abs(umin - want["argmin_I1"]) < 1e-9, (name, zmin, umin)
        cf = ax.contourf(g, g, np.minimum(Z, 0), levels=levels, cmap="Greys_r", vmin=-0.075, vmax=0.0)
        cs = ax.contour(g, g, Z, levels=levels[:-1], colors=INK2, linewidths=0.4)
        ax.clabel(cs, fmt=lambda v: "%.2f" % v, fontsize=5.8, inline_spacing=1)
        ax.plot(umin, vmin, marker="x", color="white" if zmin < -0.04 else INK, ms=6, mew=1.2)
        ax.annotate("min %.4f" % zmin, xy=(umin, vmin), xytext=(0.55 if umin < 0.5 else 0.08, 0.12 if umin < 0.5 else 0.9),
                    fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2), bbox=dict(fc="white", ec="none", pad=0.6))
        ax.set_aspect("equal"); ax.set_xlabel("$I_1$"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.text(0, 1.03, name.replace("(", " (θ = ").replace(")", ")"), fontsize=8)
        rec[name] = dict(min=r(zmin, 6), argmin=[float(umin), float(vmin)], grid_step=0.01)
    axes[0].set_ylabel("$I_2$")
    cb = fig.colorbar(cf, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("$\\sigma_C = \\hat{C}-C$ on the face $F_1=F_2=0$", fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
    save(fig, "Fig_9_1", rec)


# ---------------------------------------------------------------- Fig 11.1
def fig_11_1():
    eps = 0.3
    lo, hi = -0.6, 1.6
    fig, ax = plt.subplots(figsize=(HALF + 0.75, HALF + 0.55))
    # non-empty: T + F <= 1, T <= 1 + eps, F <= 1 + eps ; faithful: additionally T >= -eps, F >= -eps
    nonempty = [(lo, lo), (1 + eps, lo), (1 + eps, -eps), (-eps, 1 + eps), (lo, 1 + eps)]
    faithful = [(-eps, -eps), (1 + eps, -eps), (-eps, 1 + eps)]
    ax.add_patch(plt.Polygon(nonempty, closed=True, fc="white", ec=INK2, lw=0.6, hatch="....", zorder=1))
    ax.add_patch(plt.Polygon(faithful, closed=True, fc=FILL1, ec=INK, lw=1.0, zorder=2))
    ax.add_patch(plt.Polygon([(0, 0), (1, 0), (0, 1)], closed=True, fc="none", ec=INK, lw=0.9, ls="--", zorder=3))
    pts = {"(1.2, −0.3)": (1.2, -0.3), "(1.2, 0)": (1.2, 0.0)}
    ax.plot(1.2, -0.3, "o", ms=4.5, color=INK, zorder=4); ax.text(1.24, -0.36, "$(1.2, -0.3)$", fontsize=6.8, va="top")
    ax.plot(1.2, 0.0, "s", ms=4.5, mfc="white", mec=INK, zorder=4); ax.text(1.24, 0.04, "$(1.2, 0)$", fontsize=6.8, va="bottom")
    ax.text(0.25, 0.25, "faithful,\n$\\varepsilon=0.3$", fontsize=6.8, ha="center", zorder=5)
    ax.text(-0.5, -0.5, "non-empty,\nnot faithful", fontsize=6.6, ha="left", va="bottom", zorder=5,
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text(0.9, 0.9, "empty\n$(T+F>1)$", fontsize=6.8, ha="center")
    ax.annotate("$\\varepsilon=0$:\nTheorem 3", xy=(0.5, 0.5), xytext=(0.95, 1.35), fontsize=6.6,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
    ax.axhline(0, color=GRID, lw=0.5, zorder=0); ax.axvline(0, color=GRID, lw=0.5, zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("$T$"); ax.set_ylabel("$F$"); ax.set_xticks([-0.5, 0, 0.5, 1, 1.5]); ax.set_yticks([-0.5, 0, 0.5, 1, 1.5])
    # check the closed form (Proposition 8.4.1) against an LP over charges on {A, not-A}: t = Q(A) in [-eps, 1 + eps]
    from scipy.optimize import linprog
    mism, n = 0, 0
    for T in np.round(np.arange(lo, hi + 1e-9, 0.1), 10):
        for F in np.round(np.arange(lo, hi + 1e-9, 0.1), 10):
            n += 1
            bl, bu = max(T, -eps), min(1 - F, 1 + eps)
            ne_closed = bl <= bu + 1e-12
            res_lo = linprog([1], A_ub=[[-1], [1]], b_ub=[-T, 1 - F], bounds=[(-eps, 1 + eps)], method="highs")
            ne_lp = res_lo.status == 0
            faith_lp = False
            if ne_lp:
                res_hi = linprog([-1], A_ub=[[-1], [1]], b_ub=[-T, 1 - F], bounds=[(-eps, 1 + eps)], method="highs")
                L_A, L_notA = res_lo.x[0], 1 - res_hi.x[0]
                faith_lp = abs(L_A - T) < 1e-9 and abs(L_notA - F) < 1e-9
            faith_closed = ne_closed and T >= -eps - 1e-12 and F >= -eps - 1e-12 and T + F <= 1 + 1e-12 and T <= 1 + eps + 1e-12 and F <= 1 + eps + 1e-12
            mism += int(ne_lp != ne_closed) + int(faith_lp != faith_closed)
    assert mism == 0, mism
    save(fig, "Fig_11_1", dict(epsilon=eps, nonempty="T+F<=1, T<=1+eps, F<=1+eps", faithful="additionally T>=-eps, F>=-eps",
                                points=dict(faithful=[1.2, -0.3], empty=[1.2, 0.0]), lp_grid_points=n, lp_mismatches=mism))


# ---------------------------------------------------------------- Fig 12.1 (Theorem 10)
def switched_truth_lp(x1, x2, c):
    """min over joints on the 16 product atoms (marginals in K(x1), K(x2), no dependence assumption) of
    (1 - c) P(E_T^A n E_T^B) + c P(E_T^A u E_T^B)."""
    pairs = list(itertools.product(range(4), range(4)))
    P = np.array(MINIMAL, float)
    obj = [(1 - c) * (P[i, 0] * P[j, 0]) + c * max(P[i, 0], P[j, 0]) for i, j in pairs]
    A_ub, b_ub = [], []
    for k in range(3):
        A_ub.append([-P[i, k] for i, j in pairs]); b_ub.append(-x1[k])
        A_ub.append([-P[j, k] for i, j in pairs]); b_ub.append(-x2[k])
    return lp_min(np.array(obj), np.array(A_ub), np.array(b_ub), n=16)[0]


def fig_12_1():
    x1, x2 = (0.80, 0.10, 0.15), (0.60, 0.30, 0.20)     # the two centres of Exercise 13.1 (Section 11.1 setting)
    a, b = x1[0], x2[0]
    cs = np.linspace(0, 1, 201)
    T = lambda C, c: c * (a + b) + (1 - 2 * c) * C(a, b)
    ind = np.array([T(Pi, c) for c in cs]); com = np.array([T(Mc, c) for c in cs])
    na = np.array([min(T(Wc, c), T(Mc, c)) for c in cs])
    fig, ax = plt.subplots(figsize=(HALF + 1.1, 2.45))
    ax.fill_between(cs, na, np.maximum(ind, com), color=FILL2, lw=0, zorder=0)
    for k, (lab, y) in enumerate([("strong independence ($\\Pi$)", ind), ("no dependence assumption", na), ("comonotone ($M$)", com)]):
        ax.plot(cs, y, ls=LINESTYLES[k], color=SERIES[k], lw=1.2, label=lab)
    marks = []
    for c in np.linspace(0, 1, 9):
        v = switched_truth_lp(x1, x2, c)
        marks.append((float(c), v))
        assert abs(v - min(T(Wc, c), T(Mc, c))) < 1e-9
        ax.plot(c, v, marker="s", ms=4, mfc="white", mec=BLUE, ls="none", zorder=4)
    ax.axhline(0.5, color=MUTED, lw=0.5); ax.text(0.99, 0.505, "treat if $T\\geq0.5$", ha="right", va="bottom", fontsize=6.6, color=INK2)
    ax.axvline(0.5, color=MUTED, lw=0.5); ax.text(0.51, 0.31, "$c=1/2$", fontsize=6.6, color=INK2)
    # check the values of the solution of Exercise 13.1 (Table in solutions): c = 0, 0.25, 0.5
    for c, (vi, vc, vn) in {0.0: (0.48, 0.60, 0.40), 0.25: (0.59, 0.65, 0.55), 0.5: (0.70, 0.70, 0.70)}.items():
        assert abs(T(Pi, c) - vi) < 1e-9 and abs(T(Mc, c) - vc) < 1e-9 and abs(min(T(Wc, c), T(Mc, c)) - vn) < 1e-9
    ax.set_xlabel("contradiction degree $c$"); ax.set_ylabel("$T$ of the plithogenic conjunction")
    ax.set_xlim(0, 1); ax.set_ylim(0.3, 1.0); grid(ax)
    ax.legend(loc="upper left", fontsize=6.6, handlelength=2.2)
    save(fig, "Fig_10_1", dict(x1=x1, x2=x2, c_sample=r(list(cs[::25])), independence=r(list(ind[::25])),
                                comonotone=r(list(com[::25])), no_assumption=r(list(na[::25])), lp_markers=r(marks)))


# ---------------------------------------------------------------- Fig 13.1 (running example)
COURSES = {"DE": (0.5, 0.1, 0.2), "SA": (0.6, 0.2, 0.4), "FM": (0.8, 0.0, 0.1), "SM": (0.4, 0.3, 0.5)}


def grad_intervals(tr):
    l = {k: v[0] for k, v in tr.items()}; u = {k: 1 - v[2] for k, v in tr.items()}
    prod = lambda d: d["DE"] * d["SA"] * d["FM"] * d["SM"]
    sub = lambda d: min(d["DE"], d["SA"]) * min(d["FM"], d["SM"])
    return {"no assumption": (max(0.0, sum(l.values()) - 3), min(u.values())),
            "independence": (prod(l), prod(u)),
            "comonotone in subject": (sub(l), sub(u)),
            "comonotone": (min(l.values()), min(u.values()))}


def fig_13_1():
    txt = open(book_result("re05_decision.out", "expected/running_example/re05_decision.out"), encoding="utf-8").read()
    rows = ["none", "DE", "SA", "FM", "SM"]
    res = {}
    for row in rows:
        tr = dict(COURSES)
        if row != "none":
            T, I, F = tr[row]; tr[row] = (T + 0.1, I, F - 0.1)
        res[row] = grad_intervals(tr)
        m = re.search(r"^\s+%s\s+(.*)$" % row, txt.split("Step 5b")[1], re.M)
        nums = [float(x) for x in re.findall(r"[0-9]\.[0-9]{3}", m.group(1))]
        mine = [round(v, 3) for k in ("no assumption", "independence", "comonotone in subject", "comonotone") for v in res[row][k]]
        assert np.allclose(nums, mine, atol=1e-9), (row, nums, mine)
    models = ["no assumption", "independence", "comonotone in subject", "comonotone"]
    fig, ax = plt.subplots(figsize=(FULL, 2.9))
    for i, row in enumerate(rows):
        y0 = len(rows) - 1 - i
        for k, mdl in enumerate(models):
            y = y0 + 0.3 - 0.2 * k
            lo_, hi_ = res[row][mdl]
            ax.plot([lo_, hi_], [y, y], ls=LINESTYLES[k], color=SERIES[k], lw=1.6, solid_capstyle="butt",
                    label=mdl if i == 0 else None)
            ax.plot([lo_, hi_], [y, y], ls="none", marker="|", ms=5, color=SERIES[k], mew=1.0)
        if row == "SM":
            ax.add_patch(plt.Rectangle((-0.01, y0 - 0.42), 0.66, 0.84, fc="none", ec=INK2, lw=0.6, ls="--"))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(["SM + 0.1", "FM + 0.1", "SA + 0.1", "DE + 0.1", "as assessed"])
    ax.set_xlim(-0.02, 0.66); ax.set_xlabel("probability that Jenifer graduates (lower and upper)")
    ax.axvline(0.5, color=MUTED, lw=0.5)
    grid(ax, "x")
    ax.legend(loc="upper center", fontsize=6.8, ncol=4, handlelength=2.6, bbox_to_anchor=(0.5, 1.13))
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "Fig_14_1", dict(courses=COURSES, intervals={row: {m_: r(list(v), 3) for m_, v in d.items()} for row, d in res.items()},
                                checked_against="book_v11/code/running_example/re05_decision_output.txt"))


# ---------------------------------------------------------------- Fig 14.1 (Proposition 3)
def fig_14_1():
    from fig_ch02_06 import polytope_vertices, simplex_frame, tern
    P1, P2 = np.array([0.5, 0.2, 0.3]), np.array([0.1, 0.3, 0.6])
    seg = np.array([P1, P2])
    events = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    L = [min(np.dot(e, P1), np.dot(e, P2)) for e in events]     # lower probability of the segment (linear in t)
    V = polytope_vertices(events, L)
    f = np.array([10.0, 0.0, 100.0])
    seg_pay = sorted([float(f @ P1), float(f @ P2)])
    core_pay = [float(min(f @ v for v in V)), float(max(f @ v for v in V))]
    assert np.allclose(seg_pay, [35, 61]) and np.allclose(core_pay, [34, 62]), (seg_pay, core_pay)
    want = [(0.1, 0.3, 0.6), (0.5, 0.2, 0.3), (0.2, 0.2, 0.6), (0.4, 0.3, 0.3)]
    assert all(any(np.allclose(v, w) for v in V) for w in want) and len(V) == 4
    fig, ax = plt.subplots(figsize=(HALF + 0.7, HALF + 0.3))
    simplex_frame(ax, ["$1$", "$2$", "$3$"])
    ax.add_patch(plt.Polygon(tern(V), closed=True, fc=FILL2, ec=INK, lw=0.8, hatch="////", zorder=2))
    S = tern(seg)
    ax.plot(S[:, 0], S[:, 1], color=INK, lw=2.2, solid_capstyle="round", zorder=3)
    for p, lab, off in [(P1, "$P_1$", (-0.1, -0.06)), (P2, "$P_2$", (0.04, 0.0))]:
        x, y = tern(p); ax.plot(x, y, "o", ms=4, color=INK, zorder=4); ax.text(x + off[0], y + off[1], lab, fontsize=8)
    ax.annotate("core of the lower\nprobability: $E[f]\\in[%g, %g]$" % tuple(core_pay), xy=tern([0.2, 0.2, 0.6]),
                xytext=(-0.12, 0.72), fontsize=6.8, arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
    ax.annotate("segment: $E[f]\\in[%g, %g]$" % tuple(seg_pay), xy=tern(0.5 * (P1 + P2)), xytext=(0.62, 0.72), fontsize=6.8,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
    ax.set_xlim(-0.2, 1.25)
    save(fig, "Fig_7_1", dict(P1=P1.tolist(), P2=P2.tolist(), lower_probability=dict(zip(["{1}", "{2}", "{3}", "{1,2}", "{1,3}", "{2,3}"], r(L))),
                                core_vertices=r(V.tolist()), gamble=f.tolist(), payoff_segment=seg_pay, payoff_core=core_pay))


if __name__ == "__main__":
    fig_10_1(); fig_10_2(); fig_11_1(); fig_12_1(); fig_13_1(); fig_14_1()
