"""Figures of Part VI (Chapters 16-18). Every plotted number is read from\nnumbers.json (compute_numbers.py) or generality.json (generality.py), fresh copies in the output dir, else\nbook/expected/part6; the plotted values are written to figures/data/<name>.json by save().\n\nFig_16_1  one Subjective Logic opinion, three evidence states (contradiction / undetermined / ill-posed)\nFig_16_2  one Subjective Logic opinion, three triples (complete / incomplete / over-complete)\nFig_16_3  the dispute that fusion erases: SL fusion vs the RNEL tuple with fused contradiction\nFig_16_4  order dependence of sequential Definition 8.4 vs the order-free C*\nFig_17_1  architecture of the typed evidential head (diagram)\nFig_17_2  dissonance = C* + within-source part (Proposition 17.2)\nFig_17_3  experiment 23: macro-F1 by model, and mean typed components by human class\nFig_17_4  experiment 35: detection AUROC by read-out; experiment 40: vacuity vs C*\nFig_18_1  experiments 28/30: detection AUROC of RNEL-MVC vs best baseline, by scenario\nFig_18_2  experiment 28 H4 (weight on a noise-only view) and experiment 31 (query targeting vs accuracy)\n"""
import json
import os

from common import *  # noqa: F401,F403
from matplotlib.patches import FancyBboxPatch

N = json.load(open(book_result("numbers.json", "expected/part6/numbers.json"), encoding="utf-8"))
G = json.load(open(book_result("generality.json", "expected/part6/generality.json"), encoding="utf-8"))


def label_bars(ax, bars, fmt="%.2f", dy=0.01, fs=6.8):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, b.get_y() + h + dy, fmt % h, ha="center", va="bottom", fontsize=fs,
                color=INK2)


# ------------------------------------------------------------------ 16.1
def fig_16_1():
    comps = ["T", "F", "C", "U", "N", "G"]
    fills = ["white", FILL1, INK, BLUE, ORANGE, FILL2]
    hatch = ["", "///", "", "...", "xxx", ""]
    cases = [("C", "contradictory\nreports ($c=4$)"), ("U", "undetermined\nreports ($v=4$)"), ("N", "ill-posed\nreports ($n=4$)")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw=dict(width_ratios=[1, 1.6], wspace=0.55))
    # (a) SL opinion: identical
    x = np.arange(3)
    bot = np.zeros(3)
    for comp, fc, h in (("sl_b", "white", ""), ("sl_d", FILL1, "///"), ("sl_u", FILL2, "")):
        v = np.array([G["GE.%s.%s" % (c, comp)] for c, _ in cases])
        a1.bar(x, v, 0.6, bottom=bot, fc=fc, ec=INK, lw=0.5, hatch=h)
        bot += v
    a1.legend(["$b$", "$d$", "$u$"], fontsize=6.6, loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.2)
    a1.set_xticks(x, ["$c$", "$v$", "$n$"]); a1.set_ylim(0, 1.12); a1.set_ylabel("share of the evaluation")
    a1.text(1, 1.02, "$(b, d, u) = (%.1f, %.1f, %.1f)$ in all three" % (G["GE.C.sl_b"], G["GE.C.sl_d"], G["GE.C.sl_u"]),
            ha="center", va="bottom", fontsize=6.4)
    a1.set_title("(a) Subjective Logic", fontsize=8, loc="left")
    # (b) RNEL tuple: different
    bot = np.zeros(3)
    data = {}
    for comp, fc, h in zip(comps, fills, hatch):
        v = np.array([G["GE.%s.%s" % (c, comp)] for c, _ in cases])
        data[comp] = v.tolist()
        a2.bar(x, v, 0.6, bottom=bot, fc=fc, ec=INK, lw=0.5, hatch=h, label=comp)
        bot += v
    for i, (c, _) in enumerate(cases):
        a2.text(i, 1.02, "action: %s" % G["GE.%s.decision" % c], ha="center", fontsize=6.8)
    a2.set_xticks(x, [lab for _, lab in cases], fontsize=7); a2.set_ylim(0, 1.12)
    a2.legend(ncol=1, fontsize=6.8, loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.2)
    a2.set_title("(b) RNEL tuple", fontsize=8, loc="left")
    save(fig, "Fig_16_1", dict(sl={c: [G["GE.%s.sl_%s" % (c, k)] for k in "bdu"] for c, _ in cases}, tuple=data,
                                credal_interval=[G["GE.C.ci_lo"], G["GE.C.ci_hi"]]))


# ------------------------------------------------------------------ 16.2
def fig_16_2():
    names = [("complete", "complete\n(0.4, 0.2, 0.4)"), ("incomplete", "incomplete\n(0.4, 0.0, 0.4)"),
             ("overcomplete", "over-complete\n(0.6, 0.3, 0.6)")]
    trip = {"complete": (0.4, 0.2, 0.4), "incomplete": (0.4, 0.0, 0.4), "overcomplete": (0.6, 0.3, 0.6)}
    fig, ax = plt.subplots(figsize=(FULL, 2.5))
    w = 0.22
    x = np.arange(4)
    for j, (comp, fc, h) in enumerate((("T", "white", ""), ("I", FILL2, "..."), ("F", FILL1, "///"))):
        vals = [trip[k][j] for k, _ in names] + [[G["GB.complete.b"], G["GB.complete.u"], G["GB.complete.d"]][j]]
        bars = ax.bar(x + (j - 1) * w, vals, w, fc=fc, ec=INK, lw=0.5, hatch=h, label=comp)
        label_bars(ax, bars, "%.1f", fs=6.3)
    for i, (k, _) in enumerate(names):
        ax.text(i, 0.78, "sum %.1f\npara. %.1f, incompl. %.1f" % (G["GB.%s.sum" % k], G["GB.%s.paraconsistency" % k],
                                                              G["GB.%s.incompleteness" % k]),
                ha="center", fontsize=6.3, color=INK2)
    ax.axvline(2.55, color=MUTED, lw=0.6, ls="--")
    ax.set_xticks(x, [lab for _, lab in names] + ["retraction $\\nu$:\n$(b, u, d)$ of all three"], fontsize=7)
    ax.set_ylim(0, 0.95); ax.set_ylabel("value"); grid(ax)
    ax.legend(ncol=3, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 1.12))
    save(fig, "Fig_16_2", dict(triples=trip, retraction=[G["GB.complete.b"], G["GB.complete.d"], G["GB.complete.u"]],
                                paraconsistency={k: G["GB.%s.paraconsistency" % k] for k in trip}))


# ------------------------------------------------------------------ 16.3
def fig_16_3():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.5), sharey=True)
    cases = ["two sources\n(8, 0) and (0, 8)", "one source\n(8, 8)"]
    x = np.arange(2)
    sl = [N["we1.fused_b"], N["we1.fused_d"], N["we1.fused_u"]]
    bot = np.zeros(2)
    for v, fc, h, lab in zip(sl, ("white", FILL1, FILL2), ("", "///", ""), ("$b$", "$d$", "$u$")):
        a1.bar(x, [v, v], 0.55, bottom=bot, fc=fc, ec=INK, lw=0.5, hatch=h, label=lab)
        bot += v
    a1.set_xticks(x, cases, fontsize=7); a1.set_ylim(0, 1.75); a1.set_ylabel("value")
    a1.set_title("(a) Subjective Logic fusion", fontsize=8, loc="left"); a1.legend(fontsize=7, loc="upper right")
    # (b) RNEL: same T, F, G plus C for the dispute
    T, F, Gc = N["we1.T"], N["we1.fused_d"], N["we1.G"]
    Cs = [N["we1.C_def84"], N["we1.coin_C"]]
    bot = np.zeros(2)
    for v, fc, h, lab in ((np.array([T, T]), "white", "", "$T$"), (np.array([F, F]), FILL1, "///", "$F$"),
                          (np.array([Gc, Gc]), FILL2, "", "$G$"), (np.array(Cs), INK, "", "$C$")):
        a2.bar(x, v, 0.55, bottom=bot, fc=fc, ec=INK, lw=0.5, hatch=h, label=lab)
        bot += v
    for i in range(2):
        a2.text(i, bot[i] + 0.03, "total %.3f\n$C^*$ = %.3f" % (bot[i], [N["we1.Cstar"], N["we1.coin_Cstar"]][i]),
                ha="center", fontsize=6.8)
    a2.axhline(1, color=MUTED, lw=0.6, ls="--")
    a2.set_xticks(x, cases, fontsize=7); a2.set_title("(b) RNEL tuple (Definition 16.2)", fontsize=8, loc="left")
    a2.legend(fontsize=7, loc="upper right")
    save(fig, "Fig_16_3", dict(sl=sl, rnel_T_F_G=[T, F, Gc], C=Cs, Cstar=[N["we1.Cstar"], N["we1.coin_Cstar"]],
                                DC=N["we1.DC"]))


# ------------------------------------------------------------------ 16.4
def fig_16_4():
    orders = ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]
    vals = [N["we3.seq_" + o] for o in orders]
    fig, ax = plt.subplots(figsize=(HALF * 1.5, 2.3))
    ax.plot(range(6), vals, "o", color=INK, mfc="white", ms=6, label="Definition 8.4 applied sequentially")
    ax.axhline(N["we3.maxpair"], color=INK2, ls=":", lw=0.9, label="maximum over pairs")
    ax.axhline(N["we3.Cstar"], color=BLUE, ls="--", lw=1.2, label="$C^*$ (every order)")
    ax.set_xticks(range(6), orders); ax.set_xlabel("order of fusion of A = (6, 0), B = (0, 6), C = (3, 3)")
    ax.set_ylim(0.3, 0.7); ax.set_ylabel("contradiction"); grid(ax)
    ax.legend(fontsize=6.8, loc="upper right", bbox_to_anchor=(1.0, 1.02))
    save(fig, "Fig_16_4", dict(sequential=dict(zip(orders, vals)), maxpair=N["we3.maxpair"], Cstar=N["we3.Cstar"]))


# ------------------------------------------------------------------ 17.1 (diagram)
def fig_17_1():
    fig, ax = plt.subplots(figsize=(FULL, 2.3))
    xs = [0.8, 2.8, 4.8, 6.8, 8.8]
    texts = ["features $h$\n(e.g. NLI\nencoder)", "linear +\nsoftplus:\n$e_T, e_F, e_C, e_U, e_N$",
             "Def. 16.1:\ndivide by\n$\\sum e + W$", "tuple\n$(T, F, C,$\n$U, N, G)$", "policy\n(Section 16.8)"]
    BWd, BHt = 1.6, 0.9
    for x, t in zip(xs, texts):
        ax.add_patch(FancyBboxPatch((x - BWd / 2, 0.6), BWd, BHt, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec=INK, lw=0.8))
        ax.text(x, 0.6 + BHt / 2, t, ha="center", va="center", fontsize=6.6, linespacing=1.1)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(x1 - BWd / 2 - 0.02, 1.05), xytext=(x0 + BWd / 2 + 0.02, 1.05),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=8))
    ax.text(2.8, 0.35, "typed loss (Def. 17.2), labels\n$y \\in \\{T, F, C, U, N\\}$", ha="center", va="top",
            fontsize=6.2, color=INK2)
    ax.text(5.8, 0.35, "$e_C = e_U = e_N = 0$, $W = K$:\nevidential deep learning (Prop. 17.1)", ha="center",
            va="top", fontsize=6.2, color=INK2)
    ax.text(8.8, 0.35, "C, U, N, G, T/F:\none action each", ha="center", va="top", fontsize=6.2, color=INK2)
    ax.set_xlim(-0.1, 9.7); ax.set_ylim(-0.25, 1.6); ax.axis("off")
    save(fig, "Fig_17_1", dict(blocks=[t.replace("\n", " ") for t in texts]))


# ------------------------------------------------------------------ 17.2
def fig_17_2():
    names = [("dispute", "two sources\n(8,0), (0,8)"), ("coin", "one source\n(8,8)"), ("abc", "three sources\n(6,0), (0,6), (3,3)")]
    cs = [G["GK.%s.Cstar" % k] for k, _ in names]
    cw = [G["GK.%s.Cwithin" % k] for k, _ in names]
    fig, ax = plt.subplots(figsize=(HALF * 1.6, 2.4))
    x = np.arange(3)
    ax.bar(x, cs, 0.55, fc=BLUE, ec=INK, lw=0.5, label="$C^*$ (between sources)")
    ax.bar(x, cw, 0.55, bottom=cs, fc="white", ec=INK, lw=0.5, hatch="///", label="$C_w$ (within sources)")
    for i, k in enumerate(names):
        ax.text(i, cs[i] + cw[i] + 0.02, "dissonance %.3f" % G["GK.%s.dissonance" % k[0]], ha="center", fontsize=6.6)
    ax.set_xticks(x, [lab for _, lab in names], fontsize=6.8); ax.set_ylim(0, 1.15); grid(ax)
    ax.legend(fontsize=6.8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.15))
    save(fig, "Fig_17_2", dict(Cstar=cs, Cw=cw, dissonance=[G["GK.%s.dissonance" % k] for k, _ in names]))


# ------------------------------------------------------------------ 17.3
def fig_17_3():
    models = [("rnel", "RNEL head"), ("edl", "EDL"), ("softmax", "softmax"), ("logreg", "log. reg."), ("sl", "SL fusion"),
              ("rule", "rule")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    y = np.arange(len(models))
    for j, (ds, mk, fc) in enumerate((("averitec", "o", INK), ("climate", "s", "white"))):
        v = [N["e23.%s.macro.%s" % (ds, m)] for m, _ in models]
        a1.plot(v, y + (j - 0.5) * 0.18, mk, color=INK, mfc=fc, ms=5, ls="none",
                label="AVeriTeC" if ds == "averitec" else "Climate-FEVER")
    a1.set_yticks(y, [lab for _, lab in models], fontsize=7); a1.invert_yaxis()
    a1.set_xlabel("macro-F1 (nested CV, 5 seeds)"); a1.set_xlim(0.25, 0.52); grid(a1, "x")
    a1.legend(fontsize=6.8, loc="center left"); a1.set_title("(a) classification", fontsize=8, loc="left")
    # (b) mean C and N by human class, AVeriTeC
    cls = [("T", "supported"), ("C", "conflicting"), ("N", "not enough ev.")]
    x = np.arange(3)
    c = [N["e23.averitec.def81.%s.C" % k] for k, _ in cls]
    n = [N["e23.averitec.def81.%s.N" % k] for k, _ in cls]
    a2.bar(x - 0.18, c, 0.34, fc=INK, ec=INK, lw=0.5, label="mean $C$")
    a2.bar(x + 0.18, n, 0.34, fc="white", ec=INK, lw=0.5, hatch="xxx", label="mean $N$")
    a2.set_xticks(x, [lab for _, lab in cls], fontsize=6.8); a2.set_ylim(0, 0.36); grid(a2)
    a2.set_xlabel("human label (AVeriTeC)")
    a2.legend(fontsize=6.8, loc="upper left"); a2.set_title("(b) typed components", fontsize=8, loc="left")
    save(fig, "Fig_17_3", dict(macro={ds: {m: N["e23.%s.macro.%s" % (ds, m)] for m, _ in models}
                                       for ds in ("averitec", "climate")},
                                averitec_mean_C=dict(zip([k for k, _ in cls], c)),
                                averitec_mean_N=dict(zip([k for k, _ in cls], n))))


# ------------------------------------------------------------------ 17.4
def fig_17_4():
    ro = [("R1_Cstar", "$C^*$"), ("B4_TMC_K", "Dempster conflict"), ("B2_dissonance", "fused dissonance"),
          ("B5_plurality", "plurality"), ("R2_Def84_maxpair", "Def. 8.4 max-pair"), ("B3_RCML_DC", "RCML DC"),
          ("B1_vacuity", "fused vacuity")]
    ds = [("HandWritten", "o", INK), ("Scene", "s", "white"), ("PIE", "^", BLUE)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.7), gridspec_kw=dict(width_ratios=[1.6, 1]))
    y = np.arange(len(ro))
    data = {}
    for j, (d, mk, fc) in enumerate(ds):
        v = [N["e35.%s.auroc.%s" % (d, k)] for k, _ in ro]
        data[d] = dict(zip([k for k, _ in ro], v))
        a1.plot(v, y + (j - 1) * 0.2, mk, color=INK, mfc=fc, ms=4.5, ls="none", label=d)
    a1.axvline(0.5, color=MUTED, lw=0.6, ls="--")
    a1.set_yticks(y, [lab for _, lab in ro], fontsize=7); a1.invert_yaxis(); a1.set_xlim(0.4, 1.0)
    a1.set_xlabel("detection AUROC (exp. 35)"); grid(a1, "x"); a1.legend(fontsize=6.6, loc="lower right")
    a1.set_title("(a) multi-view conflict", fontsize=8, loc="left")
    ks = ["k=4", "k=6", "k=10"]
    u = [G["GJ.e40.%s.auroc_u" % k] for k in ks]
    cs = [G["GJ.e40.%s.auroc_Cstar" % k] for k in ks]
    x = np.arange(3)
    a2.bar(x - 0.18, u, 0.34, fc="white", ec=INK, lw=0.5, hatch="///", label="vacuity $u$")
    a2.bar(x + 0.18, cs, 0.34, fc=BLUE, ec=INK, lw=0.5, label="$C^*$")
    a2.set_xticks(x, ["%s sources" % k[2:] for k in ks], fontsize=6.8); a2.set_ylim(0, 1.15); grid(a2)
    a2.set_ylabel("AUROC, conflict vs rest"); a2.legend(fontsize=6.6, loc="upper left", ncol=2)
    a2.set_title("(b) synthetic streams (exp. 40)", fontsize=8, loc="left")
    save(fig, "Fig_17_4", dict(e35=data, e40=dict(u=u, Cstar=cs)))


# ------------------------------------------------------------------ 18.1
def fig_18_1():
    sc = [("S1", "S1 view swap"), ("S2", "S2 sensor failure"), ("S8", "S8 mixed swaps"), ("S5", "S5 drift"),
          ("S6", "S6 partial corruption")]
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.5), sharey=True)
    data = {}
    for ax, ds, title in ((axs[0], "dsa", "(a) DSA"), (axs[1], "har", "(b) HAR")):
        y = np.arange(len(sc))
        r = [N["e30.%s.%s.rnel" % (ds, s)] for s, _ in sc]
        b = [N["e30.%s.%s.best" % (ds, s)] for s, _ in sc]
        data[ds] = dict(rnel=dict(zip([s for s, _ in sc], r)), best=dict(zip([s for s, _ in sc], b)),
                        best_name={s: N["e30.%s.%s.best_name" % (ds, s)] for s, _ in sc})
        for i in range(len(sc)):
            ax.plot([b[i], r[i]], [i, i], color=MUTED, lw=1)
        ax.plot(b, y, "s", color=INK, mfc="white", ms=5, label="best baseline")
        ax.plot(r, y, "o", color=INK, mfc=BLUE, ms=5, label="RNEL-MVC")
        ax.set_yticks(y, [lab for _, lab in sc], fontsize=7); ax.invert_yaxis(); ax.set_xlim(0.45, 1.0)
        ax.set_xlabel("detection AUROC"); grid(ax, "x"); ax.set_title(title, fontsize=8, loc="left")
    axs[1].legend(fontsize=6.8, loc="lower left")
    save(fig, "Fig_18_1", data)


# ------------------------------------------------------------------ 18.2
def fig_18_2():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.5), gridspec_kw=dict(wspace=0.45))
    x = np.arange(2)
    rn = [N["e28.dsa.H4.weight"], N["e28.har.H4.weight"]]
    wm = [N["e28.dsa.S4.wmvec_weight"], N["e28.har.S4.wmvec_weight"]]
    un = [N["e28.dsa.H4.uniform"], N["e28.har.H4.uniform"]]
    a1.bar(x - 0.18, rn, 0.34, fc=BLUE, ec=INK, lw=0.5, label="RNEL-MVC")
    a1.bar(x + 0.18, wm, 0.34, fc="white", ec=INK, lw=0.5, hatch="///", label="WMVEC")
    for i in range(2):
        a1.plot([i - 0.4, i + 0.4], [un[i], un[i]], color=INK, ls="--", lw=0.8, label="uniform" if i == 0 else None)
    a1.set_xticks(x, ["DSA (5 views)", "HAR"]); a1.set_ylabel("weight of the noise-only view"); a1.set_ylim(0, 0.62)
    a1.legend(fontsize=6.8, loc="upper left"); grid(a1); a1.set_title("(a) corrupted view (exp. 28, H4)", fontsize=8, loc="left")
    sc = ["S1", "S2", "S8"]
    for ds, mk, fc in (("dsa", "o", BLUE), ("har", "s", "white")):
        q = [N["e31.%s.AL1.%s" % (ds, s)] for s in sc]
        acc = [N["e31.%s.AL2.%s" % (ds, s)] for s in sc]
        a2.plot(q, acc, mk, color=INK, mfc=fc, ms=5, ls="none", label=ds.upper())
        for s, xx, yy in zip(sc, q, acc):
            a2.text(xx + 0.008, yy, s, fontsize=6.3, va="center")
    a2.axhline(0, color=MUTED, lw=0.6); a2.axvline(0, color=MUTED, lw=0.6)
    a2.set_xlim(-0.42, 0.05); a2.set_ylim(-0.065, 0.015)
    a2.set_xlabel("change in share of contaminated queries", fontsize=8); a2.set_ylabel("change in clean AULC", fontsize=8)
    a2.legend(fontsize=6.8, loc="lower right"); a2.set_title("(b) active learning (exp. 31)", fontsize=8, loc="left")
    save(fig, "Fig_18_2", dict(H4=dict(rnel=rn, wmvec=wm, uniform=un),
                                AL={ds: {s: [N["e31.%s.AL1.%s" % (ds, s)], N["e31.%s.AL2.%s" % (ds, s)]] for s in sc}
                                    for ds in ("dsa", "har")}))


if __name__ == "__main__":
    for f in (fig_16_1, fig_16_2, fig_16_3, fig_16_4, fig_17_1, fig_17_2, fig_17_3, fig_17_4, fig_18_1, fig_18_2):
        f()
