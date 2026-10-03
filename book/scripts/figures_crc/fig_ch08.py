"""Figures of Chapter 8 of the proposed plan (glut lifting and N-norms as natural extensions; Section 7 of v12/v13).

Fig_6_1  Three frames for one triple: the classical frame {x, not-x}, the Belnap frame {t, f, b, n} and the minimal
         glut lifting (Theorem 8). Membership of each atom in E_T, E_I, E_F, and what each frame does with the triple
         x = (0.8, 0.3, 0.6) (linear programming).
Fig_6_2  Faithful regions of the disjoint, Belnap and minimal frames (Theorem 8(d)) on two slices I = 0.2 and I = 0.6,
         computed by linear programming on a grid and compared with the closed-form regions.
Fig_6_3  The three N-norms as natural extensions on the product of two minimal liftings (Theorem 9): closed forms
         (lines) against the natural extension computed numerically (markers): products of extreme points for strong
         independence and a linear programme on the 16 atoms with no dependence assumption.
"""
from common import *  # noqa: F401,F403
import itertools
from scipy.optimize import linprog

X0 = (0.8, 0.3, 0.6)


def k_nonempty_and_envelope(patterns, x):
    env = lifting_envelope(patterns, x)
    return env


def min_event(patterns, x, member):
    """min P(event) over K(x); member: list of 0/1 per atom."""
    P = np.array(patterns, float)
    v, _ = lp_min(np.array(member, float), -P.T, -np.array(x), n=len(P))
    return v


def fig_8_1():
    frames = [
        ("(a) classical frame", [("$x$", (1, None, 0)), ("$\\bar{x}$", (0, None, 1))]),
        ("(b) Belnap frame", [("$t$", (1, 0, 0)), ("$f$", (0, 0, 1)), ("$b$", (1, 0, 1)), ("$n$", (0, 1, 0))]),
        ("(c) minimal glut lifting", [("$(0,1,1)$", (0, 1, 1)), ("$(1,0,1)$", (1, 0, 1)), ("$(1,1,0)$", (1, 1, 0)),
                                      ("$(1,1,1)$", (1, 1, 1))]),
    ]
    # numbers for the triple X0
    T, I, F = X0
    delta_classical = max(0.0, (T + F - 1) / 2)
    bel = lifting_envelope(BELNAP, X0)
    mini = lifting_envelope(MINIMAL, X0)
    glut_min = min_event(MINIMAL, X0, [1 if (p[0] and p[2]) else 0 for p in MINIMAL])
    assert mini is not None and np.allclose(mini, X0, atol=1e-9)
    assert abs(glut_min - (T + F - 1)) < 1e-9
    notes = ["$[T, 1-F] = [0.8, 0.4]$: empty\nsure loss $\\delta = %.1f$; $I$ not used" % delta_classical,
             ("$K(x)=\\emptyset$" if bel is None else "envelope $(%.2f, %.2f, %.2f)$" % tuple(bel)) +
             "\nfaithful only if $T+I\\leq1$, $F+I\\leq1$",
             "envelope $= (0.8, 0.3, 0.6)$: faithful\nmass on gluts $\\geq T+F-1 = %.1f$" % glut_min]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.35))
    for ax, (ttl, atoms), note in zip(axes, frames, notes):
        n = len(atoms)
        for j, ev in enumerate(["$E_T$", "$E_I$", "$E_F$"]):
            ax.text(j, n - 0.35, ev, ha="center", va="bottom", fontsize=8)
        for i, (lab, pat) in enumerate(atoms):
            y = n - 1 - i
            glut = pat[0] == 1 and pat[2] == 1
            if glut:
                ax.add_patch(plt.Rectangle((-0.5, y - 0.45), 3.0, 0.9, fc="white", ec="none", hatch="....", zorder=0))
            ax.text(-0.65, y, lab, ha="right", va="center", fontsize=7.6)
            for j, bit in enumerate(pat):
                if bit is None:
                    ax.text(j, y, "–", ha="center", va="center", fontsize=8, color=INK2)
                    continue
                ax.add_patch(plt.Rectangle((j - 0.28, y - 0.28), 0.56, 0.56, fc=INK if bit else "white",
                                           ec=INK, lw=0.6, zorder=2))
        ax.set_xlim(-1.9, 2.6); ax.set_ylim(-1.75, n + 0.05)
        ax.axis("off")
        ax.text(-1.9, n + 0.35, ttl, fontsize=7.8, ha="left")
        ax.text(-1.85, -0.75, note, fontsize=6.5, ha="left", va="top", linespacing=1.2)
    save(fig, "Fig_6_1", dict(triple=X0, classical_delta=delta_classical,
                               belnap_envelope=None if bel is None else r(bel), minimal_envelope=r(mini),
                               min_glut_mass_minimal=r(glut_min),
                               patterns=dict(belnap=BELNAP, minimal=MINIMAL)))


def fig_8_2():
    grid_ = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(1, 2, figsize=(FULL - 0.5, 2.4), sharey=True)
    rec = {}
    for ax, I in zip(axes, (0.2, 0.6)):
        mism = 0
        for name, pats, fc, h in [("minimal", MINIMAL, "white", ""), ("Belnap", BELNAP, FILL2, ""),
                                  ("disjoint", DISJOINT, FILL1, "////")]:
            ok = np.zeros((len(grid_), len(grid_)), bool)
            for a, T in enumerate(grid_):
                for b, F in enumerate(grid_):
                    env = lifting_envelope(pats, (T, I, F))
                    ok[b, a] = env is not None and np.allclose(env, (T, I, F), atol=1e-7)
                    closed = {"minimal": True, "Belnap": T + I <= 1 + 1e-12 and F + I <= 1 + 1e-12,
                              "disjoint": T + I + F <= 1 + 1e-12}[name]
                    mism += int(ok[b, a] != closed)
            rec.setdefault("I=%.1f" % I, {})[name] = dict(share_faithful=round(float(ok.mean()), 4))
            # draw the closed-form region (polygon) that the LP grid confirms
            if name == "minimal":
                poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
            elif name == "Belnap":
                poly = [(0, 0), (1 - I, 0), (1 - I, 1 - I), (0, 1 - I)]
            else:
                poly = [(0, 0), (1 - I, 0), (0, 1 - I)]
            ax.add_patch(plt.Polygon(poly, closed=True, fc=fc, ec=INK, lw=0.8, hatch=h, zorder={"minimal": 1, "Belnap": 2, "disjoint": 3}[name]))
        rec["I=%.1f" % I]["grid_points"] = len(grid_) ** 2
        rec["I=%.1f" % I]["lp_vs_closed_form_mismatches"] = mism
        assert mism == 0, (I, mism)
        ax.plot([0, 1], [1, 0], color=INK, lw=0.6, ls="--", zorder=4)
        ax.text(0.97, 0.97, "paraconsistent\nzone $T+F>1$", ha="right", va="top", fontsize=6.6, zorder=5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("$T$"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.text(0.0, 1.04, "$I = %.1f$" % I, fontsize=8)
    axes[0].set_ylabel("$F$")
    from matplotlib.patches import Patch
    fig.legend([Patch(fc=FILL1, ec=INK, hatch="////", lw=0.6), Patch(fc=FILL2, ec=INK, lw=0.6), Patch(fc="white", ec=INK, lw=0.6)],
               ["disjoint frame: $T+I+F\\leq 1$", "Belnap frame: $T+I\\leq1$, $F+I\\leq1$", "minimal glut lifting: whole square"],
               loc="center left", bbox_to_anchor=(0.80, 0.5), fontsize=7, handlelength=1.5)
    fig.subplots_adjust(right=0.78, wspace=0.12)
    save(fig, "Fig_6_2", rec)


# ---------------------------------------------------------------- Theorem 9 on the product of two minimal liftings
def k_vertices(x):
    """Extreme points of K(x) on the minimal lifting (enumerated by active constraints)."""
    P = np.array(MINIMAL, float)
    rows = [(P[:, k], x[k]) for k in range(3)] + [(np.eye(4)[i], 0.0) for i in range(4)]
    V = []
    for comb in itertools.combinations(rows, 3):
        A = np.vstack([c[0] for c in comb] + [np.ones(4)])
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        p = np.linalg.solve(A, [c[1] for c in comb] + [1.0])
        if (p >= -1e-12).all() and all(P[:, k] @ p >= x[k] - 1e-12 for k in range(3)):
            if not any(np.allclose(p, q) for q in V):
                V.append(p)
    return V


def conj_events():
    """Indicators on the 16 product atoms of the conjunction events: E_T^A n E_T^B, E_I^A u E_I^B, E_F^A u E_F^B."""
    pairs = list(itertools.product(range(4), range(4)))
    P = MINIMAL
    eT = [1.0 if P[i][0] and P[j][0] else 0.0 for i, j in pairs]
    eI = [1.0 if P[i][1] or P[j][1] else 0.0 for i, j in pairs]
    eF = [1.0 if P[i][2] or P[j][2] else 0.0 for i, j in pairs]
    return pairs, [eT, eI, eF]


def ne_independent(x1, x2):
    V1, V2 = k_vertices(x1), k_vertices(x2)
    pairs, ev = conj_events()
    out = []
    for e in ev:
        out.append(min(sum(e[k] * v1[i] * v2[j] for k, (i, j) in enumerate(pairs)) for v1 in V1 for v2 in V2))
    return out


def ne_no_assumption(x1, x2):
    pairs, ev = conj_events()
    P = np.array(MINIMAL, float)
    A_ub, b_ub = [], []
    for k in range(3):
        A_ub.append([-P[i, k] for i, j in pairs]); b_ub.append(-x1[k])
        A_ub.append([-P[j, k] for i, j in pairs]); b_ub.append(-x2[k])
    return [lp_min(np.array(e), np.array(A_ub), np.array(b_ub), n=16)[0] for e in ev]


def fig_8_3():
    x2 = (0.6, 0.3, 0.5)          # a non-normalised second conjunct (sum 1.4)
    ts = np.linspace(0, 1, 101)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.4))
    # (a) truth of the conjunction as T1 varies, I1 = 0.2, F1 = 0.4
    T2 = x2[0]
    curves = {"independence: $T_1T_2$": ts * T2, "no assumption: $\\max(0, T_1+T_2-1)$": np.maximum(0, ts + T2 - 1),
              "comonotone: $\\min(T_1, T_2)$": np.minimum(ts, T2)}
    for k, (lab, y) in enumerate(curves.items()):
        a1.plot(ts, y, ls=LINESTYLES[k], color=SERIES[k], lw=1.2, label=lab)
    # (b) indeterminacy as I1 varies, T1 = 0.5, F1 = 0.4
    I2 = x2[1]
    cI = {"independence: $I_1+I_2-I_1I_2$": ts + I2 - ts * I2, "no assumption, comonotone: $\\max(I_1, I_2)$": np.maximum(ts, I2)}
    for k, (lab, y) in enumerate(cI.items()):
        a2.plot(ts, y, ls=LINESTYLES[k], color=SERIES[k], lw=1.2, label=lab)
    # markers: natural extension computed numerically at a few points
    pts = np.linspace(0, 1, 6)
    errs = dict(independence=0.0, no_assumption=0.0)
    mk = {"ind": [], "na": []}
    for t in pts:
        x1 = (t, 0.2, 0.4)
        ni = ne_independent(x1, x2); na = ne_no_assumption(x1, x2)
        errs["independence"] = max(errs["independence"], abs(ni[0] - t * T2), abs(ni[1] - (0.2 + I2 - 0.2 * I2)), abs(ni[2] - (0.4 + 0.5 - 0.2)))
        errs["no_assumption"] = max(errs["no_assumption"], abs(na[0] - max(0, t + T2 - 1)), abs(na[1] - max(0.2, I2)), abs(na[2] - max(0.4, 0.5)))
        mk["ind"].append((t, ni[0])); mk["na"].append((t, na[0]))
        a1.plot(t, ni[0], marker=MARKERS[0], ms=4.5, mfc="white", mec=SERIES[0], ls="none")
        a1.plot(t, na[0], marker=MARKERS[1], ms=4.5, mfc="white", mec=SERIES[1], ls="none")
        x1b = (0.5, t, 0.4)
        nib = ne_independent(x1b, x2); nab = ne_no_assumption(x1b, x2)
        errs["independence"] = max(errs["independence"], abs(nib[1] - (t + I2 - t * I2)))
        errs["no_assumption"] = max(errs["no_assumption"], abs(nab[1] - max(t, I2)))
        a2.plot(t, nib[1], marker=MARKERS[0], ms=4.5, mfc="white", mec=SERIES[0], ls="none")
        a2.plot(t, nab[1], marker=MARKERS[1], ms=4.5, mfc="white", mec=SERIES[1], ls="none")
    assert max(errs.values()) < 1e-9, errs
    for ax, xl, yl, tag in [(a1, "$T_1$", "$T$ of the conjunction", "(a)"), (a2, "$I_1$", "$I$ of the conjunction", "(b)")]:
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); grid(ax, "both")
        ax.legend(loc="upper left", fontsize=6.4, handlelength=2.2, borderaxespad=0.2)
        ax.text(1.0, 0.02, tag, ha="right", va="bottom", fontsize=8)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker="o", mfc="white", mec=INK, ls="none"), Line2D([], [], marker="s", mfc="white", mec=BLUE, ls="none")],
               ["numerical: products of extreme points", "numerical: linear programme on 16 atoms"],
               loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=7)
    fig.subplots_adjust(bottom=0.27, wspace=0.3)
    save(fig, "Fig_6_3", dict(second_conjunct=x2, panel_a_first_conjunct="(T1, 0.2, 0.4)", panel_b_first_conjunct="(0.5, I1, 0.4)",
                               marker_points=r(list(pts)), max_error_vs_closed_forms=errs,
                               markers_truth=dict(independence=r(mk["ind"]), no_assumption=r(mk["na"]))))


if __name__ == "__main__":
    fig_8_1(); fig_8_2(); fig_8_3()
