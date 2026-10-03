"""Figures of Chapters 2-6 of the proposed plan (Foundations, Preliminaries, Reduction, Classical frame).

Fig_2_1  The simplex of probabilities on three outcomes: (a) the credal set of Ellsberg's urn (Example F1);
         (b) the credal set and natural extension of Example F4. Vertices and bounds computed by linear programming and
         checked against book_v12/results/foundations_examples.json (Tables F.1).
Fig_2_2  The neutrosophic cube [0,1]^3: the normalised plane T + I + F = 1 and the regions below (incomplete) and above
         (paraconsistent) it; the plane T + F = 1 that bounds the zone of sure loss of the betting reading (Theorem 3).
Fig_3_1  The betting reading (T, I, F) -> [T, 1 - F] (Definition 2, Theorems 1, 3, 4) for normalised and
         non-normalised triples.
Fig_3_2  How thin coherence is: random normalised dual assignments classified by Walley's conditions (Table 2),
         read from book_v12/code/results/t1_reduction_hierarchy.json.
Fig_4_1  The (T, F) plane of one event: avoiding sure loss, the degree of sure loss (T + F - 1)/2, and off values
         (Theorem 3), with an LP check of the degree on a grid.
"""
from common import *  # noqa: F401,F403
import itertools
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

S3 = math.sqrt(3) / 2


def tern(p):
    """Barycentric (p1, p2, p3) -> 2-D: vertex 1 at (0, 0), vertex 2 at (1, 0), vertex 3 at (1/2, sqrt3/2)."""
    p = np.asarray(p, dtype=float)
    return np.array([p[..., 1] + 0.5 * p[..., 2], S3 * p[..., 2]]).T


def polytope_vertices(A, b):
    """Vertices of {p in simplex_3 : A p >= b} by enumeration of pairs of active constraints."""
    rows = [(np.array(a, float), bb) for a, bb in zip(A, b)] + [(np.eye(3)[i], 0.0) for i in range(3)]
    V = []
    for (a1, b1), (a2, b2) in itertools.combinations(rows, 2):
        M_ = np.vstack([a1, a2, np.ones(3)])
        if abs(np.linalg.det(M_)) < 1e-12:
            continue
        p = np.linalg.solve(M_, [b1, b2, 1.0])
        if all(np.dot(a, p) >= bb - 1e-9 for a, bb in rows):
            if not any(np.allclose(p, q) for q in V):
                V.append(p)
    c = np.mean(V, axis=0)
    V.sort(key=lambda p: math.atan2(*(tern(p) - tern(c))[::-1]))
    return np.array(V)


def simplex_frame(ax, labels):
    tri = tern(np.eye(3))
    ax.add_patch(plt.Polygon(tri, closed=True, fc="white", ec=INK2, lw=0.7))
    for k in range(1, 10):          # hairline grid
        t = k / 10
        for i in range(3):
            a = np.zeros(3); a[i] = t; a[(i + 1) % 3] = 1 - t
            b = np.zeros(3); b[i] = t; b[(i + 2) % 3] = 1 - t
            P = tern(np.array([a, b]))
            ax.plot(P[:, 0], P[:, 1], color=GRID, lw=0.4, zorder=0)
    pos = [(-0.04, -0.04, "right", "top"), (1.04, -0.04, "left", "top"), (0.5, S3 + 0.04, "center", "bottom")]
    for (x, y, ha, va), lab in zip(pos, labels):
        ax.text(x, y, lab, ha=ha, va=va, fontsize=8.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.12, S3 + 0.12)


def fig_2_1():
    R = json.load(open(book_result("foundations_examples.json", "expected/foundations/foundations_examples.json")))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.55))
    # (a) Ellsberg: P(R) = 1/3, P(B) + P(Y) = 2/3 -> segment {(1/3, b, 2/3 - b)}
    simplex_frame(a1, ["$R$", "$B$", "$Y$"])
    seg = np.array([[1 / 3, 0, 2 / 3], [1 / 3, 2 / 3, 0]])
    S = tern(seg)
    a1.plot(S[:, 0], S[:, 1], color=INK, lw=2.2, solid_capstyle="round", zorder=3)
    a1.plot(*tern([1 / 3, 1 / 3, 1 / 3]), "o", ms=4.5, mfc="white", mec=INK, zorder=4)
    a1.annotate("$P(R)=1/3$,\n$P(B)\\in[0, 2/3]$", xy=tern([1 / 3, 0.45, 2 / 3 - 0.45]), xytext=(0.80, 0.62),
                fontsize=7.5, ha="left", arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
    a1.text(0.02, S3 + 0.07, "(a)", fontsize=8.5, transform=a1.transData)
    lb = {"black": R["ellsberg"]["black"], "yellow": R["ellsberg"]["yellow"], "red_or_black": R["ellsberg"]["red_or_black"]}
    # check the segment against the recorded bounds
    assert abs(lb["black"][1] - 2 / 3) < 1e-6 and abs(lb["black"][0]) < 1e-9
    # (b) Example F4: L({1}) = 0.2, L({2}) = 0.3, L({1,2}) = 0.6
    simplex_frame(a2, ["$1$", "$2$", "$3$"])
    A = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]; b = [0.2, 0.3, 0.6]
    V = polytope_vertices(A, b)
    a2.add_patch(plt.Polygon(tern(V), closed=True, fc=FILL1, ec=INK, lw=1.0, hatch="///", zorder=2))
    # the three constraint lines, dashed: p1 = 0.2, p2 = 0.3, p1 + p2 = 0.6 (i.e. p3 = 0.4)
    for P0, P1, lab, xy in [([0.2, 0.8, 0], [0.2, 0, 0.8], r"$P(\{1\})=0.2$", (0.17, 0.52)),
                            ([0.7, 0.3, 0], [0, 0.3, 0.7], r"$P(\{2\})=0.3$", (0.73, 0.52)),
                            ([0.6, 0, 0.4], [0, 0.6, 0.4], r"$P(\{1,2\})=0.6$", (0.5, 0.39))]:
        Q = tern(np.array([P0, P1]))
        a2.plot(Q[:, 0], Q[:, 1], ls="--", lw=0.6, color=INK2, zorder=1)
    # natural extension by LP and check against Table F.1
    events = {"[0]": [1, 0, 0], "[1]": [0, 1, 0], "[2]": [0, 0, 1], "[0, 1]": [1, 1, 0], "[0, 2]": [1, 0, 1], "[1, 2]": [0, 1, 1]}
    natext = {}
    for k, e in events.items():
        lo, _ = lp_min(np.array(e, float), -np.array(A, float), -np.array(b))
        hi, _ = lp_min(-np.array(e, float), -np.array(A, float), -np.array(b))
        natext[k] = [lo, -hi]
        assert abs(natext[k][0] - R["natext"][k][0]) < 1e-6 and abs(natext[k][1] - R["natext"][k][1]) < 1e-6, k
    # vertex labels
    for v in V:
        x, y = tern(v)
        a2.plot(x, y, "o", ms=3, color=INK, zorder=3)
    a2.annotate("$\\mathcal{M}(L)$", xy=tern(np.mean(V, axis=0)), xytext=(0.84, 0.58), fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
    a2.text(0.02, S3 + 0.07, "(b)", fontsize=8.5)
    save(fig, "Fig_2_1", dict(ellsberg_segment=r(seg.tolist()), ellsberg_bounds_recorded=lb,
                               F4_vertices=r(V.tolist()), F4_natural_extension=r([natext[k] for k in events]),
                               F4_events=list(events)))


def fig_3_1():
    fig = plt.figure(figsize=(HALF + 0.6, HALF + 0.45))
    ax = fig.add_subplot(projection="3d")
    ax.set_proj_type("ortho")

    def P(t, i, f):        # axes: x = T, y = F, z = I (so that the plane T + F = 1 is vertical)
        return (t, f, i)
    for s_, e_ in itertools.combinations(itertools.product([0, 1], repeat=3), 2):
        if sum(abs(np.array(s_) - np.array(e_))) == 1:
            ax.plot(*zip(s_, e_), color=INK2, lw=0.5)
    tri = [P(1, 0, 0), P(0, 1, 0), P(0, 0, 1)]
    ax.add_collection3d(Poly3DCollection([tri], facecolor=FILL1, edgecolor=INK, lw=0.9, alpha=0.55))
    q = [P(1, 0, 0), P(1, 1, 0), P(0, 1, 1), P(0, 0, 1)]
    xs, ys, zs = zip(*(q + [q[0]]))
    ax.plot(xs, ys, zs, color=INK, lw=0.8, ls="--")
    ax.scatter(*P(0, 0, 0), s=9, color=INK, depthshade=False)
    ax.scatter(*P(1, 1, 1), s=9, color=INK, depthshade=False)
    ax.scatter(*P(0.6, 0.3, 0.2), s=12, marker="s", color=INK, depthshade=False)
    ax.set_xlabel("$T$", labelpad=-10); ax.set_ylabel("$F$", labelpad=-10); ax.set_zlabel("$I$", labelpad=-10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.tick_params(pad=-4, labelsize=6.5)
    ax.view_init(elev=22, azim=-25)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor("none"); axis._axinfo["grid"]["linewidth"] = 0
    ax.set_box_aspect((1, 1, 1))
    # labels in screen coordinates, with leader lines to points of the regions
    from mpl_toolkits.mplot3d import proj3d
    fig.canvas.draw()

    def scr(pt):
        x2, y2, _ = proj3d.proj_transform(*pt, ax.get_proj())
        return ax.transData.transform((x2, y2))
    inv = fig.transFigure.inverted()
    for pt, txt, xy in [(P(0.15, 0.1, 0.1), "$T+I+F<1$\n(incomplete)", (0.02, 0.10)),
                        (P(0.85, 0.85, 0.75), "$T+I+F>1$\n(paraconsistent)", (0.70, 0.93)),
                        (P(0.6, 0.3, 0.2), "$(0.6, 0.3, 0.2)$", (0.70, 0.12)),
                        (P(0.25, 0.95, 0.75), "$T+F=1$", (0.02, 0.93))]:
        fx, fy = inv.transform(scr(pt))
        fig.add_artist(plt.Line2D([fx, xy[0] + 0.05], [fy, xy[1] + (0.04 if xy[1] < 0.5 else -0.02)],
                                  lw=0.45, color=INK2, transform=fig.transFigure))
        fig.text(xy[0], xy[1], txt, fontsize=6.8, ha="left", va="center", bbox=dict(fc="white", ec="none", pad=0.5))
    rng = np.random.default_rng(20261003)
    X = rng.random((200000, 3)); s = X.sum(1)
    shares = dict(below=float((s < 1).mean()), above=float((s > 1).mean()), exact_below=1 / 6,
                  sure_loss_zone_T_plus_F_gt_1=float((X[:, 0] + X[:, 2] > 1).mean()))
    assert abs(shares["below"] - 1 / 6) < 0.005
    save(fig, "Fig_2_2", dict(plane="T+I+F=1 (shaded)", sure_loss_boundary="T+F=1 (dashed)", axes="x=T, y=F, z=I",
                               example_triple=[0.6, 0.3, 0.2], uniform_shares_of_cube=shares))


def fig_4_1():
    rows = [  # (label, (T, I, F), group)
        ("$(0.5, 0.3, 0.2)$", (0.5, 0.3, 0.2), "a"),
        ("$(0.2, 0.6, 0.2)$", (0.2, 0.6, 0.2), "a"),
        ("$(0.7, 0.1, 0.2)$", (0.7, 0.1, 0.2), "a"),
        ("$(0.3, 0.1, 0.4)$", (0.3, 0.1, 0.4), "b"),
        ("$(0.3, 0.6, 0.4)$", (0.3, 0.6, 0.4), "b"),
        ("$(0.7, 0.2, 0.5)$", (0.7, 0.2, 0.5), "b"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.9), sharex=True)
    rec = []
    for ax, g, ttl in zip(axes, "ab", ["(a) normalised: width $=I$", "(b) not normalised: $I$ not used"]):
        sub = [x for x in rows if x[2] == g]
        for j, (lab, (T, I, F), _) in enumerate(sub):
            y = len(sub) - 1 - j
            lo, hi = T, 1 - F
            ax.plot([0, 1], [y, y], color=GRID, lw=0.6, zorder=0)
            if lo <= hi:
                ax.plot([lo, hi], [y, y], color=INK, lw=2.6, solid_capstyle="butt", zorder=2)
                ax.plot([lo, lo], [y - 0.18, y + 0.18], color=INK, lw=0.9); ax.plot([hi, hi], [y - 0.18, y + 0.18], color=INK, lw=0.9)
                note = "$[%.1f, %.1f]$" % (lo, hi)
            else:
                delta = (T + F - 1) / 2
                ax.add_patch(plt.Rectangle((hi, y - 0.16), lo - hi, 0.32, fc="white", ec=INK, lw=0.7, hatch="xxxx", zorder=2))
                note = "empty; $\\delta=%.2f$" % delta
            ax.text(1.03, y, note, va="center", fontsize=7)
            ax.text(-0.03, y, lab, va="center", ha="right", fontsize=7.2)
            rec.append(dict(triple=[T, I, F], interval=[lo, hi], nonempty=lo <= hi + 1e-12,
                            width=round(hi - lo, 4), I=I, delta=round(max(0.0, (T + F - 1) / 2), 4)))
        ax.set_ylim(-0.6, len(sub) - 0.4); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel("$P(A)$")
        ax.text(0.0, len(sub) - 0.25, ttl, fontsize=7.6, ha="left")
    for x in rec:
        if abs(sum(x["triple"]) - 1) < 1e-9:
            assert abs(x["width"] - x["I"]) < 1e-9          # Theorem 1: I = U - L on the normalised plane
    fig.subplots_adjust(wspace=0.95)
    save(fig, "Fig_3_1", dict(rows=rec))


def fig_4_2():
    R = json.load(open(book_result("t1_reduction_hierarchy.json", "expected/dossier/t1_reduction_hierarchy.json")))["random_normalized_dual"]
    ns = sorted(R, key=int)
    cats = [("coherent", "coherent", "white", ""), ("asl_incoherent", "avoids sure loss, incoherent", FILL1, "///"),
            ("sure_loss", "sure loss", INK2, "")]
    fig, ax = plt.subplots(figsize=(FULL, 1.55))
    rec = {}
    for j, n in enumerate(ns):
        y = len(ns) - 1 - j
        tot = R[n]["samples"]; left = 0
        rec[n] = {}
        for key, lab, fc, h in cats:
            w = R[n][key] / tot
            rec[n][key] = dict(count=R[n][key], share=round(w, 4))
            if w > 0:
                ax.barh(y, w, left=left, height=0.56, color=fc, edgecolor=INK, lw=0.6, hatch=h)
            left += w
        c = R[n]["coherent"] / tot
        ax.text(1.015, y, "%.1f %% coherent\n(%d samples)" % (100 * c, tot), va="center", fontsize=7)
    from matplotlib.patches import Patch
    ax.legend([Patch(fc=fc, ec=INK, lw=0.6, hatch=h) for _, _, fc, h in cats], [lab for _, lab, _, _ in cats],
              loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=3, fontsize=7, handlelength=1.6)
    assert rec["3"]["coherent"]["count"] == 38 and rec["4"]["coherent"]["count"] == 0 and rec["2"]["coherent"]["share"] == 1
    ax.set_yticks(range(len(ns))); ax.set_yticklabels(["$n=%s$" % n for n in ns[::-1]])
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("share of random normalised dual assignments")
    save(fig, "Fig_3_2", rec)


def fig_6_1():
    fig, ax = plt.subplots(figsize=(HALF + 0.55, HALF + 0.35))
    lo, hi = -0.25, 1.3
    # zone of sure loss within the unit square and off values outside it
    ax.add_patch(plt.Polygon([(1, 0), (1, 1), (0, 1)], closed=True, fc=FILL2, ec="none", zorder=0))
    tt = np.linspace(0, 1, 201)
    TT, FF = np.meshgrid(tt, tt)
    D = np.maximum(0, (TT + FF - 1) / 2)
    cs = ax.contour(TT, FF, D, levels=[0.1, 0.2, 0.3, 0.4], colors=INK2, linewidths=0.5)
    for v in (0.1, 0.2, 0.3, 0.4):     # label each level line just above the square, on its end point F = 1
        ax.text(2 * v, 1.025, "%.1f" % v, ha="center", va="bottom", fontsize=6.3, color=INK2)
    ax.text(0.95, 1.025, "$\\delta$", ha="left", va="bottom", fontsize=7, color=INK2)
    ax.plot([0, 1], [1, 0], color=INK, lw=1.1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="none", ec=INK, lw=0.8))
    # off values: overset T > 1 (sure loss), underset T < 0 (corrected to 0)
    ax.add_patch(plt.Rectangle((1, lo), hi - 1, hi - lo, fc="white", ec="none", hatch="////", zorder=0))
    ax.add_patch(plt.Rectangle((lo, lo), -lo, hi - lo, fc="white", ec="none", hatch="....", zorder=0))
    ax.text(0.25, 0.25, "avoids\nsure loss", ha="center", va="center", fontsize=7.3)
    ax.text(0.80, 0.56, "sure loss\n$T+F>1$", ha="center", va="center", fontsize=7.3,
            bbox=dict(fc=FILL2, ec="none", pad=1.2))
    ax.text(1.15, 0.5, "overset $T>1$:\nsure loss", ha="center", va="center", fontsize=6.5, rotation=90,
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text(-0.125, 0.5, "underset $T<0$:\ncorrected to 0", ha="center", va="center", fontsize=6.5, rotation=90,
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_xlim(lo, hi); ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("$T$"); ax.set_ylabel("$F$")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_aspect("equal")
    # LP check of delta on a grid (Theorem 3(b)): relax both bounds by eps; minimise eps
    from scipy.optimize import linprog
    worst = 0.0
    for T in np.linspace(0, 1, 11):
        for F in np.linspace(0, 1, 11):
            # variables p (=P(A)), eps; minimise eps s.t. p >= T - eps, 1 - p >= F - eps, 0 <= p <= 1, eps >= 0
            res = linprog([0, 1], A_ub=[[-1, -1], [1, -1]], b_ub=[-T, 1 - F], bounds=[(0, 1), (0, None)], method="highs")
            worst = max(worst, abs(res.x[1] - max(0, (T + F - 1) / 2)))
    assert worst < 1e-9
    save(fig, "Fig_4_1", dict(contours=[0.1, 0.2, 0.3, 0.4], formula="delta=(T+F-1)^+/2", lp_grid_points=121,
                               lp_max_error=worst))


if __name__ == "__main__":
    fig_2_1(); fig_3_1(); fig_4_1(); fig_4_2(); fig_6_1()
