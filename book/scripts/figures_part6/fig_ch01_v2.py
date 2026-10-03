"""Fig_1_1_v2: map of the exact readings of neutrosophic evidence (replaces Figure 1.1 of the CRC edition).\n\nEvery edge names the result that proves it. Changes with respect to crc_assets/figures/scripts/fig_ch01.py:\nthe central node is the triple as a state of evidence; the IDM edge reads "IDM extended by vacuous completion";\nthe learning readings (RNEL tuple, evidential deep learning, multi-view clustering) are added (Part VI).\n"""
from common import *  # noqa: F401,F403
from matplotlib.patches import FancyBboxPatch

CX = [1.15, 4.65, 8.15]
RY = [9.3, 7.3, 5.2, 3.1, 1.0]
NODES = {
    "ST": (CX[0], RY[0], "Refined neutrosophic\nestimates $a+\\sum_k b_k I_k$"),
    "MV": (CX[1], RY[0], "Multi-view clustering:\nconflict between views"),
    "ED": (CX[2], RY[0], "Evidential deep learning\n$(b_1,\\dots,b_K,u)$"),
    "PI": (CX[0], RY[1], "Probability intervals;\nimprecise Dirichlet model"),
    "CR": (CX[1], RY[1], "Conjugate lower/upper\nprobabilities $[L, U]$"),
    "RN": (CX[2], RY[1], "RNEL evidence tuple\n$(T,C,U,N,G,F,I_k)$"),
    "SL": (CX[0], RY[2], "Subjective Logic\nopinions $(b, d, u)$"),
    "NP": (CX[1], RY[2], "Neutrosophic triple\n$(T, I, F)$ as evidence"),
    "OFF": (CX[2], RY[2], "Signed credal sets\n(off-values)"),
    "BD": (CX[0], RY[4], "Belnap–Dunn frame\n$\\{t, f, b, n\\}$"),
    "GL": (CX[1], RY[3], "Credal sets on the\nminimal glut lifting"),
    "DS": (CX[2], RY[3], "Hybrid DSm model\n(DSmT)"),
    "PL": (CX[2], RY[4], "Plithogenic\nprobability"),
    "LOG": (CX[1], RY[4], "Probability logics\nof $T$, $I$, $F$"),
    "PR": (CX[0], RY[3], "Probability\n$I = 0$, $T + F = 1$"),
}
EDGES = [
    ("ST", "PI", "special", "Prop. 12.7.1: range = IDM\nextended by vacuous completion"),
    ("PI", "CR", "special", "Thm 2, Cor 2"),
    ("CR", "NP", "exact", "Thm 1: bijection under\n(N), (D), (B); $I = U - L$"),
    ("SL", "NP", "special", "Thm 5;\nProp. 16.2"),
    ("NP", "OFF", "lift", "Thm 8.4.2"),
    ("NP", "GL", "lift", "Thm 8: whole cube,\n$m+1$ atoms, unique"),
    ("BD", "GL", "lift", "Thm 8(d);\nProp. 16.3"),
    ("GL", "DS", "exact", "§12.1"),
    ("PL", "GL", "lift", "Cor 3"),
    ("GL", "LOG", "exact", "Thms 12.6.1–12.6.6:\ncompleteness"),
    ("RN", "NP", "lift", "Defs 16.1–16.2"),
    ("ED", "RN", "special", "Prop. 17.1"),
    ("RN", "MV", "lift", "Ch. 18"),
    ("PR", "SL", "special", "$u = 0$;\nProp. 16.1"),
]
NW, NH = 2.3, 0.9


def clip(x0, y0, x1, y1):
    """Point where the segment from the centre (x0, y0) towards (x1, y1) leaves the box around (x0, y0)."""
    dx, dy = x1 - x0, y1 - y0
    t = min(NW / 2 / abs(dx) if dx else 1e9, NH / 2 / abs(dy) if dy else 1e9)
    return x0 + t * dx, y0 + t * dy


def fig():
    fig, ax = plt.subplots(figsize=(FULL, 6.3))
    for k, (x, y, t) in NODES.items():
        main = k in ("NP", "RN")
        ax.add_patch(FancyBboxPatch((x - NW / 2, y - NH / 2), NW, NH, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=FILL2 if main else "white", ec=INK, lw=1.4 if main else 0.7, zorder=2))
        ax.text(x, y, t, ha="center", va="center", fontsize=6.8, zorder=3, linespacing=1.15)
    ls = {"exact": "-", "special": "--", "lift": ":"}
    for a, b, kind, lab in EDGES:
        xa, ya, _ = NODES[a]; xb, yb, _ = NODES[b]
        if abs(ya - yb) < 1e-9:
            sgn = 1 if xb > xa else -1
            p0, p1 = (xa + sgn * NW / 2, ya), (xb - sgn * NW / 2, yb)
            lx, ly, ha, va = (p0[0] + p1[0]) / 2, ya + 0.1, "center", "bottom"
        elif abs(xa - xb) < 1e-9:
            sgn = 1 if yb > ya else -1
            p0, p1 = (xa, ya + sgn * NH / 2), (xb, yb - sgn * NH / 2)
            lx, ly, ha, va = xa + 0.1, (p0[1] + p1[1]) / 2, "left", "center"
        else:  # diagonal: clip at the box edges
            p0, p1 = clip(xa, ya, xb, yb), clip(xb, yb, xa, ya)
            lx, ly, ha, va = 0.62 * p0[0] + 0.38 * p1[0], 0.62 * p0[1] + 0.38 * p1[1], "center", "center"
        ax.annotate("", xy=p1, xytext=p0, zorder=1,
                    arrowprops=dict(arrowstyle="<|-|>" if kind == "exact" else "-|>", lw=0.8, ls=ls[kind],
                                    color=INK, shrinkA=0, shrinkB=0, mutation_scale=8))
        ax.text(lx, ly, lab, fontsize=6.1, ha=ha, va=va, color=INK2, zorder=4, linespacing=1.1,
                bbox=dict(fc="white", ec="none", pad=0.6))
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=INK, ls="-", lw=0.8), Line2D([], [], color=INK, ls="--", lw=0.8),
          Line2D([], [], color=INK, ls=":", lw=0.8)]
    ax.legend(hs, ["exact correspondence", "special case of (arrow points to the more general)",
                   "represented on a larger frame / computed from"],
              loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=6.8, handlelength=2.6)
    ax.set_xlim(-0.1, 9.4); ax.set_ylim(RY[-1] - 0.6, RY[0] + 0.6); ax.axis("off")
    save(fig, "Fig_1_1_v2", dict(nodes={k: v[2].replace("\n", " ") for k, v in NODES.items()},
                                  edges=[dict(source=a, target=b, kind=s, result=l.replace("\n", " "))
                                         for a, b, s, l in EDGES]))


if __name__ == "__main__":
    fig()


# ---------------------------------------------------------------- Fig_1_2_v2: dependency chart of the 19 chapters
CH19 = {1: "Introduction", 2: "Foundations", 3: "Normalised\nsubclass, IDM", 4: "Classical\nframe", 5: "Glut lifting,\nN-norms",
        6: "Imprecise\nprobability", 7: "Subjective\nLogic", 8: "Probability\nlogics", 9: "Indeterminacy\nprofiles",
        10: "Choosing an\nN-norm", 11: "Off-values", 12: "Plithogeny,\napplications", 13: "RNEL, fused\ninterval",
        14: "Refined\nstatistics", 15: "Running\nexample, rnel", 16: "RNEL logic", 17: "Evidential\nlearning",
        18: "Multi-view\nclustering", 19: "Questions,\nagenda"}
ROWS = [("I", [1, 2]), ("II", [3, 4, 5, 6]), ("III", [7, 8]), ("IV", [9, 10, 11, 12]), ("V", [13, 14, 15]),
        ("VI", [16, 17, 18]), ("", [19])]
DEPS19 = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (4, 7), (5, 8), (5, 9), (5, 11), (5, 12), (9, 10), (7, 13), (13, 14),
          (14, 15), (13, 16), (16, 17), (16, 18), (16, 19)]
# Chapter 6 uses Chapters 3-5 (drawn as 5 -> 6); Chapter 19 also uses Chapters 8, 11 and 14; those arrows are not drawn, to keep the chart readable.
CORE19 = {2, 3, 4, 5, 12}
COL = [0.9, 2.7, 4.5, 6.3, 8.1]
POS19 = {1: (COL[1], 0), 2: (COL[2], 0), 3: (COL[0], 1), 4: (COL[1], 1), 5: (COL[2], 1), 6: (COL[3], 1),
         7: (COL[0], 2), 8: (COL[4], 2), 10: (COL[1], 3), 9: (COL[2], 3), 11: (COL[3], 3), 12: (COL[4], 3),
         13: (COL[0], 4), 14: (COL[1], 4), 15: (COL[2], 4), 16: (COL[0], 5), 17: (COL[1], 5), 18: (COL[2], 5),
         19: (COL[3], 6)}
ARC19 = {(16, 18): 0.35}


def fig_1_2():
    fig, ax = plt.subplots(figsize=(FULL, 5.4))
    pos = {c: (x, 7.0 - 1.05 * r) for c, (x, r) in POS19.items()}
    for r, (part, chs) in enumerate(ROWS):
        ax.text(-0.25, 7.0 - 1.05 * r, part, fontsize=8, fontweight="bold", va="center", ha="left")
    bw, bh = 1.6, 0.6
    for a, b in DEPS19:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        if abs(y1 - y2) < 1e-9 and (a, b) not in ARC19:
            sg = 1 if x2 > x1 else -1
            p0, p1 = (x1 + sg * bw / 2, y1), (x2 - sg * bw / 2, y2)
        elif (a, b) in ARC19:
            e = -bh / 2 if ARC19[(a, b)] > 0 else bh / 2
            p0, p1 = (x1, y1 + e), (x2, y2 + e)
        else:
            p0, p1 = (x1, y1 - bh / 2), (x2, y2 + bh / 2)
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", lw=0.55, color=INK2, mutation_scale=7,
                    shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=%g" % ARC19.get((a, b), 0.0)), zorder=1)
    for c, (x, y) in pos.items():
        part = [p for p, chs in ROWS if c in chs][0]
        fc = FILL2 if part in ("II", "IV", "VI") else "white"
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=fc, ec=INK, lw=1.5 if c in CORE19 else 0.6, zorder=2))
        ax.text(x - bw / 2 + 0.06, y, str(c), ha="left", va="center", fontsize=7.0, fontweight="bold", zorder=3)
        ax.text(x - bw / 2 + 0.34, y, CH19[c], ha="left", va="center", fontsize=6.1, zorder=3, linespacing=1.0)
    ax.set_xlim(-0.35, 9.0); ax.set_ylim(7.0 - 1.05 * 6 - 0.5, 7.45); ax.axis("off")
    save(fig, "Fig_1_2_v2", dict(chapters={k: v.replace("\n", " ") for k, v in CH19.items()}, prerequisites=DEPS19,
                                  core_course=sorted(CORE19), parts={p: chs for p, chs in ROWS}))


fig_1_2()
