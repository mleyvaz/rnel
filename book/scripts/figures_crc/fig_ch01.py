"""Chapter 1 (and Preface) figures.

Fig_1_2  Dependency chart of the 16 chapters of the CRC edition (CHAPTER_PLAN.md); the arrows are the
         prerequisites listed in the plan (data: CHAPTERS and DEPS below, the same table as CHAPTER_PLAN.md).
Fig_1_1  Map of the exact correspondences proved in the book (a diagram; every edge names the result that proves it).
"""
from common import *  # noqa: F401,F403
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------- chapter plan (mirrors CHAPTER_PLAN.md, 16 chapters)
CHAPTERS = {
    1: "Introduction", 2: "Foundations", 3: "Normalised\nsubclass", 4: "The classical\nframe",
    5: "Subjective\nLogic", 6: "Glut lifting\nand N-norms", 7: "Imprecise\nprobability",
    8: "Indeterminacy\nprofiles", 9: "Choosing\nan N-norm", 10: "Plithogeny,\napplications",
    11: "Off-values", 12: "RNEL, fused\ninterval", 13: "Refined\nstatistics",
    14: "Running\nexample, rnel", 15: "Probability\nlogics", 16: "Discussion\nand agenda",
}
DEPS = [(2, 3), (3, 4), (3, 7), (4, 5), (4, 6), (6, 8), (6, 10), (6, 11), (8, 9), (5, 12), (12, 13), (10, 14),
        (13, 15), (11, 15), (15, 16)]
# Direct prerequisites of CHAPTER_PLAN.md; prerequisites implied by transitivity, and the use of the imprecise
# Dirichlet model (Chapter 3) in Chapters 13 and 14, are listed in the plan but not drawn, to keep the chart readable.
CORE = {2, 3, 4, 6, 10}                 # one-semester course (Preface)
PART = {1: "I", 2: "I", 3: "II", 4: "II", 5: "II", 6: "II", 7: "II", 8: "III", 9: "III", 10: "III", 11: "III",
        12: "IV", 13: "IV", 14: "IV", 15: "V", 16: "V"}
X = [0.9, 2.75, 4.6, 6.45, 8.3]
Y = [7.0 - 0.8 * k for k in range(7)]
POS = {1: (X[1], Y[0]), 2: (X[2], Y[0]),
       3: (X[2], Y[1]),
       4: (X[2], Y[2]), 7: (X[4], Y[2]),
       6: (X[2], Y[3]), 5: (X[3], Y[3]),
       8: (X[0], Y[4]), 10: (X[1], Y[4]), 12: (X[3], Y[4]), 11: (X[4], Y[4]),
       9: (X[0], Y[5]), 14: (X[1], Y[5]), 13: (X[3], Y[5]), 15: (X[4], Y[5]),
       16: (X[4], Y[6])}
ARC = {(3, 7): -0.12}
PART_FILL = {"I": "#ffffff", "II": FILL2, "III": "#ffffff", "IV": FILL2, "V": "#ffffff"}
BW, BH = 1.66, 0.5


def box(ax, xy, k):
    x, y = xy
    core = k in CORE
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=PART_FILL[PART[k]], ec=INK, lw=1.5 if core else 0.6, zorder=2))
    ax.text(x - BW / 2 + 0.06, y, str(k), ha="left", va="center", fontsize=7.2, fontweight="bold", zorder=3)
    ax.text(x - BW / 2 + 0.30, y, CHAPTERS[k], ha="left", va="center", fontsize=6.3, zorder=3, linespacing=1.0)


def edge_points(a, b):
    (x1, y1), (x2, y2) = POS[a], POS[b]
    if abs(y1 - y2) < 1e-9:
        return (x1 + BW / 2, y1), (x2 - BW / 2, y2)
    return (x1, y1 - BH / 2), (x2, y2 + BH / 2)


def fig_1_1():
    fig, ax = plt.subplots(figsize=(FULL, 4.3))
    for a, b in DEPS:
        p0, p1 = edge_points(a, b)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.6, color=INK2, zorder=1,
                                     connectionstyle="arc3,rad=%g" % ARC.get((a, b), 0.0), shrinkA=1, shrinkB=1))
    for k in POS:
        box(ax, POS[k], k)
    # legend
    leg = [("white", 0.6, "Parts I, III, V"), (FILL2, 0.6, "Parts II, IV"), ("white", 1.5, "one-semester course")]
    lx, ly = 0.75, Y[1] + 0.25
    for i, (fc, lw, lab) in enumerate(leg):
        yy = ly - 0.26 * i
        ax.add_patch(FancyBboxPatch((lx - 0.42, yy - 0.08), 0.28, 0.16, boxstyle="round,pad=0.01", fc=fc, ec=INK, lw=lw))
        ax.text(lx - 0.07, yy, lab, va="center", fontsize=6.8)
    ax.set_xlim(0.0, 9.2); ax.set_ylim(Y[-1] - 0.35, Y[0] + 0.35); ax.axis("off")
    save(fig, "Fig_1_2", dict(chapters={k: v.replace("\n", " ") for k, v in CHAPTERS.items()}, prerequisites=DEPS,
                               core_course=sorted(CORE), parts=PART))


# ---------------------------------------------------------------- Fig 1.2: map of the exact correspondences
CX = [1.15, 4.65, 8.15]
RY = [8.2, 6.2, 4.1, 2.0, -0.1]
NODES = {
    "ST": (CX[0], RY[0], "Refined neutrosophic\nestimates $a+\\sum b_k I_k$"),
    "PI": (CX[0], RY[1], "Probability intervals;\nimprecise Dirichlet model"),
    "CR": (CX[1], RY[1], "Conjugate lower/upper\nprobabilities $[L, U]$"),
    "BF": (CX[2], RY[1], "Belief functions;\n2-monotone capacities"),
    "SL": (CX[0], RY[2], "Subjective Logic\nopinions $(b, d, u)$"),
    "NP": (CX[1], RY[2], "Neutrosophic probability\n$(T, I, F)$ on events"),
    "OFF": (CX[2], RY[2], "Signed credal sets\n(off-values)"),
    "BD": (CX[0], RY[3], "Belnap–Dunn frame\n$\\{t, f, b, n\\}$"),
    "GL": (CX[1], RY[3], "Credal sets on the\nminimal glut lifting"),
    "DS": (CX[2], RY[3], "Hybrid DSm model\n(DSmT)"),
    "PL": (CX[0], RY[4], "Plithogenic\nprobability"),
    "LOG": (CX[1], RY[4], "Probability logics\nof $T$, $I$, $F$"),
}
EDGES = [  # (a, b, kind, label): exact = solid two-headed; special = dashed (a is a special case / part of b);
    #                               lift = dotted (b represents a on a larger frame)
    ("ST", "PI", "special", "Prop. 12.7.1: range = IDM\nextended by vacuous completion"),
    ("PI", "CR", "special", "Thm 2, Cor 2"),
    ("BF", "CR", "special", "Cor 1"),
    ("CR", "NP", "exact", "Thm 1: bijection under\n(N), (D), (B); $I = U - L$"),
    ("SL", "NP", "special", "Thm 5"),
    ("NP", "OFF", "lift", "Thm 8.4.2"),
    ("NP", "GL", "lift", "Thm 8: whole cube,\n$m+1$ atoms, unique"),
    ("BD", "GL", "lift", "Thm 8(d)"),
    ("GL", "DS", "exact", "§12.1"),
    ("PL", "GL", "lift", "Cor 3"),
    ("GL", "LOG", "exact", "Thms 12.6.1–12.6.4:\ncomplete axiomatisations"),
]
NW, NH = 2.3, 0.9


def fig_1_2():
    fig, ax = plt.subplots(figsize=(FULL, 5.2))
    for k, (x, y, t) in NODES.items():
        ax.add_patch(FancyBboxPatch((x - NW / 2, y - NH / 2), NW, NH, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=FILL2 if k == "NP" else "white", ec=INK, lw=1.4 if k == "NP" else 0.7, zorder=2))
        ax.text(x, y, t, ha="center", va="center", fontsize=7.0, zorder=3, linespacing=1.15)
    ls = {"exact": "-", "special": "--", "lift": ":"}
    for a, b, kind, lab in EDGES:
        xa, ya, _ = NODES[a]; xb, yb, _ = NODES[b]
        if abs(ya - yb) < 1e-9:          # horizontal edge
            sgn = 1 if xb > xa else -1
            p0, p1 = (xa + sgn * NW / 2, ya), (xb - sgn * NW / 2, yb)
            lx, ly, ha, va = (p0[0] + p1[0]) / 2, ya + 0.1, "center", "bottom"
        else:                              # vertical edge
            sgn = 1 if yb > ya else -1
            p0, p1 = (xa, ya + sgn * NH / 2), (xb, yb - sgn * NH / 2)
            lx, ly, ha, va = xa + 0.1, (p0[1] + p1[1]) / 2, "left", "center"
        ax.annotate("", xy=p1, xytext=p0, zorder=1,
                    arrowprops=dict(arrowstyle="<|-|>" if kind == "exact" else "-|>", lw=0.8, ls=ls[kind],
                                    color=INK, shrinkA=0, shrinkB=0, mutation_scale=8))
        ax.text(lx, ly, lab, fontsize=6.4, ha=ha, va=va, color=INK2, zorder=4, linespacing=1.1)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=INK, ls="-", lw=0.8), Line2D([], [], color=INK, ls="--", lw=0.8),
          Line2D([], [], color=INK, ls=":", lw=0.8)]
    ax.legend(hs, ["exact correspondence", "special case or part of", "represented on a larger frame"],
              loc="upper left", bbox_to_anchor=(0.43, 0.995), ncol=1, fontsize=7, handlelength=2.6)
    ax.set_xlim(-0.1, 9.4); ax.set_ylim(RY[-1] - 0.6, RY[0] + 0.6); ax.axis("off")
    save(fig, "Fig_1_1", dict(nodes={k: v[2].replace("\n", " ") for k, v in NODES.items()},
                               edges=[dict(source=a, target=b, kind=s, result=l.replace("\n", " "))
                                      for a, b, s, l in EDGES]))


if __name__ == "__main__":
    fig_1_1()
    fig_1_2()
