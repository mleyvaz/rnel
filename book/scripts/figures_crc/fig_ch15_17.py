"""Figures of Chapters 15-17 of the proposed plan.

Fig_12_1  Experiment 36 (Section 9 / 9.1.3): AUROC with 95 % intervals for detecting human-labelled conflict and the
          errors of the fused classifier, AVeriTeC and Climate-FEVER; read from experiments/36_.../summary_*.json.
Fig_12_2  (a) Memory needed to keep the between-source conflict (experiment 37): the evidential state (R, S, Kw) per
          class against the credal set over the sources, 8 bytes per number, checked against
          experiments/37_.../summary.json; (b) experiment 38, RNEL-stream minus mean fusion (Table 9.1.4), read from
          experiments/38_.../hypotheses.json.
Fig_13_2  Width decomposition of the two-source example (Section 12.7.5) by named type and by source (rnel.neutro_stats).
Fig_13_1  p(x) + p(not x) in the two-source example: exact 1 with the symbols kept, interval arithmetic, glut, gap.
Fig_13_3  The acetaminophen/ASD case (Section 12.7.7): range of p(P) and composition of its width by year,
          read from neutro_case_paracetamol/results/cumulative.csv and recomputed with rnel for 2026.
Fig_13_4  The same case: naive cumulative random-effects ratio by year against the identification-design estimates,
          read from results/a4_classical_cumulative.csv and results/summary.json (forest-style).
Fig_13_5  Simulation with known truth (Section 12.7.8): mean absolute error against the prevalence of confounding.
Fig_13_6  Simulation: worst-case regret over the five scenarios, computed from sim_summary.csv and checked against
          simulation/results/posthoc_regret.txt.
Fig_15_1  The translation route to completeness (Section 12.6): logics, semantics and the results that connect them.
"""
from common import *  # noqa: F401,F403
import csv
import re
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

from rnel.neutro_stats import estimate, estimate_sources, decompose, interval_sum, credal_interval  # noqa: E402
from rnel.tuple import Reports  # noqa: E402


# ---------------------------------------------------------------- Fig 15.1
def fig_15_1():
    scores = [("C*_K", "$C^{\\star}$ (multiclass)"), ("width_src", "width, credal set over sources"),
              ("width_hull", "width, fused interval"), ("entropy", "entropy"), ("margin", "margin")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL - 0.2, 2.45), sharey=True)
    rec = {}
    for ax, (ds, name) in zip(axes, [("averitec", "AVeriTeC"), ("climate", "Climate-FEVER")]):
        s = json.load(open(EXP + "/36_credal_vs_neutro_real_conflict/results/summary_%s.json" % ds))
        rec[ds] = {}
        for j, (k, lab) in enumerate(scores):
            y = len(scores) - 1 - j
            for t, (task, mk, col, dy) in enumerate([("task1_conflict", "o", INK, 0.13), ("task2_error", "s", BLUE, -0.13)]):
                m = s[task][k]["auroc_mean"]; lo, hi = s[task][k]["auroc_ci95"]
                ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=0.9)
                ax.plot(m, y + dy, marker=mk, ms=4.2, mfc="white" if t else col, mec=col, ls="none")
                rec[ds].setdefault(k, {})[task] = dict(auroc=r(m, 3), ci95=r([lo, hi], 3))
        ax.axvline(0.5, color=MUTED, lw=0.6)
        ax.text(0.5, len(scores) - 0.45, "chance", fontsize=6.5, color=INK2, ha="center")
        ax.set_xlim(0.35, 0.8); ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8]); ax.set_xlabel("AUROC (mean of 5 seeds, 95 % interval)")
        ax.text(0.35, len(scores) - 0.2, name, fontsize=8)
        grid(ax, "x")
    # quoted values of Section 9.1 (verify_v12_numbers.py)
    assert rec["averitec"]["width_src"]["task1_conflict"]["auroc"] == 0.652 and rec["averitec"]["C*_K"]["task1_conflict"]["auroc"] == 0.588
    assert rec["averitec"]["entropy"]["task2_error"]["auroc"] == 0.736 and rec["climate"]["entropy"]["task2_error"]["auroc"] == 0.634
    axes[0].set_yticks(range(len(scores))); axes[0].set_yticklabels([lab for _, lab in scores][::-1], fontsize=7.5)
    axes[0].tick_params(axis="y", length=0)
    fig.legend([Line2D([], [], marker="o", color=INK, mfc=INK, lw=0.9), Line2D([], [], marker="s", color=BLUE, mfc="white", lw=0.9)],
               ["task 1: detect human-labelled conflict", "task 2: detect errors of the fused classifier"],
               loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=2, fontsize=7)
    fig.subplots_adjust(bottom=0.27, wspace=0.08)
    save(fig, "Fig_12_1", rec)


# ---------------------------------------------------------------- Fig 15.2
def fig_15_2():
    S = json.load(open(EXP + "/37_neutro_vs_credal_economy/results/summary.json"))
    H = json.load(open(EXP + "/38_conflict_aware_streaming_fusion/results/hypotheses.json"))
    K = 20
    Vs = np.logspace(1, 4, 61)
    state = np.full_like(Vs, 3 * K * 8.0)
    credal = Vs * K * 8.0
    assert S["memory_K20_V10000"]["R1"] == 3 * K * 8 and S["memory_K20_V10000"]["C1_vertices"] == 10000 * K * 8
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.5), gridspec_kw=dict(width_ratios=[1, 1.25]))
    a1.loglog(Vs, credal, color=BLUE, ls="--", lw=1.3, label="credal set over the sources ($VK$ numbers)")
    a1.loglog(Vs, state, color=INK, ls="-", lw=1.3, label="evidential state $(R, S, K_w)$ ($3K$ numbers)")
    a1.plot([10000], [S["memory_K20_V10000"]["C1_vertices"]], "s", mfc="white", mec=BLUE, ms=4.5)
    a1.plot([10000], [S["memory_K20_V10000"]["R1"]], "o", mfc="white", mec=INK, ms=4.5)
    a1.text(9000, S["memory_K20_V10000"]["C1_vertices"] * 1.6, "1.6 MB", ha="right", fontsize=6.8)
    a1.text(9000, S["memory_K20_V10000"]["R1"] * 1.6, "480 B", ha="right", fontsize=6.8)
    a1.set_xlabel("number of sources $V$ ($K = 20$ classes)"); a1.set_ylabel("memory (bytes)")
    a1.set_ylim(1e2, 1e7)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=6.3, handlelength=2.2)
    a1.text(0.0, 1.02, "(a)", transform=a1.transAxes, va="bottom", fontsize=8)
    # (b) experiment 38, regime P, rho = 0.3
    row = lambda ds, comp, cond: [x for x in H[ds][comp] if x["cond"] == cond][0]
    dsets = [("har", "HAR"), ("Scene", "Scene15"), ("PIE", "PIE")]
    faults = [("swap", "o"), ("disconnection", "s"), ("drift", "^")]
    rec = {}
    for i, (ds, dname) in enumerate(dsets):
        for j, (f, mk) in enumerate(faults):
            x = row(ds, "rnel_stream_vs_mean|P|0.3", "%s_P_0.3" % f)
            o = row(ds, "oracle_vs_mean|P|0.3", "%s_P_0.3" % f)
            y = len(dsets) - 1 - i + 0.22 - 0.22 * j
            a2.plot(x["ci"], [y, y], color=SERIES[j], lw=0.9)
            a2.plot(x["diff"], y, marker=mk, ms=4.3, color=SERIES[j], mfc="white" if j else SERIES[j], ls="none",
                    label=f if i == 0 else None)
            a2.plot(o["diff"], y, marker="|", ms=6, color=INK2, ls="none", label="oracle (removes the faulty source)" if (i == 0 and j == 0) else None)
            rec.setdefault(dname, {})[f] = dict(diff=r(x["diff"]), ci=r(x["ci"]), oracle=r(o["diff"]))
    assert rec["HAR"]["swap"]["diff"] == 0.0156 and rec["Scene15"]["disconnection"]["diff"] == 0.0143
    a2.axvline(0, color=MUTED, lw=0.6)
    a2.set_yticks(range(len(dsets))); a2.set_yticklabels([d for _, d in dsets][::-1]); a2.tick_params(axis="y", length=0)
    a2.set_xlabel("RNEL-stream minus mean fusion (accuracy)")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=6.3, handlelength=1.2, ncol=2)
    a2.text(0.0, 1.02, "(b)", transform=a2.transAxes, va="bottom", fontsize=8)
    grid(a2, "x")
    fig.subplots_adjust(wspace=0.35, bottom=0.36)
    save(fig, "Fig_12_2", dict(memory_K20_V10000=S["memory_K20_V10000"], bytes_per_number=8, exp38_regime_P_rho_0_3=rec))


# ---------------------------------------------------------------- the two-source example (rnel.neutro_stats demo)
W = 2.0
SRC = {"trial A": Reports(t=4, c=2), "cohort B": Reports(f=3, c=1, n=2)}
POOLED = Reports(t=4, f=3, c=3, n=2)


def fig_16_1():
    e = estimate_sources(SRC, W)
    lo, hi = credal_interval(POOLED, W)
    bt, bs = decompose(e.x, "type"), decompose(e.x, "source")
    assert abs((hi - lo) - e.x.width) < 1e-12 and abs(sum(bt.values()) - e.x.width) < 1e-12
    names_t = {"C": "contradiction $I_C$", "N": "ill-posed $I_N$", "G": "prior weight $I_G$"}
    names_s = {"trial A": "source A", "cohort B": "source B", "G": "no source (prior)"}
    fig, ax = plt.subplots(figsize=(FULL, 1.9))
    rows = [("range of $p(x)$", None), ("by type", bt), ("by source", bs)]
    hatches = ["", "////", "....", "xxxx"]
    fills = [FILL1, "white", "white", FILL2]
    rec = {}
    for i, (lab, d) in enumerate(rows):
        y = 1.3 * (len(rows) - 1 - i)
        if d is None:
            ax.barh(y, hi - lo, left=lo, height=0.5, color=INK2, edgecolor=INK, lw=0.6)
            ax.text(lo - 0.01, y, "%.3f" % lo, ha="right", va="center", fontsize=7)
            ax.text(hi + 0.01, y, "%.3f" % hi, ha="left", va="center", fontsize=7)
            continue
        left = lo
        rec[lab] = {}
        keys = ["C", "N", "G"] if lab == "by type" else ["trial A", "cohort B", "G"]
        for k, key in enumerate(keys):
            w = d.get(key, 0.0)
            nm = (names_t if lab == "by type" else names_s)[key]
            ax.barh(y, w, left=left, height=0.5, color=fills[k], edgecolor=INK, lw=0.6, hatch=hatches[k])
            ax.text(left + w / 2, y - 0.38, "%s\n%.3f" % (nm, w), ha="center", va="top", fontsize=6.5, linespacing=1.0)
            rec[lab][nm] = r(w)
            left += w
    ax.set_yticks([1.3 * k for k in range(len(rows))]); ax.set_yticklabels([x[0] for x in rows][::-1]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1); ax.set_ylim(-1.0, 1.3 * (len(rows) - 1) + 0.45); ax.set_xlabel("probability scale")
    ax.spines["left"].set_visible(False)
    grid(ax, "x")
    save(fig, "Fig_13_2", dict(S=4 + 3 + 3 + 2 + W, credal_interval=r([lo, hi]), width=r(hi - lo), decomposition=rec,
                                coefficients={str(k): r(v) for k, v in e.x.terms.items()}))


def fig_16_2():
    e = estimate_sources(SRC, W)
    ia = interval_sum(e.x, e.not_x)
    g = estimate_sources(SRC, W, glut=True); p = estimate_sources(SRC, W, gap=True); gb = estimate_sources(SRC, W, glut=True, gap=True)
    vals = [("symbols kept (credal reading)", e.total.a, e.total.width),
            ("interval arithmetic on the ranges", None, ia),
            ("glut reading: $1 + c/S$", g.total.a, None),
            ("gap reading: $1 - n/S$", p.total.a, None),
            ("glut and gap: $1 + (c - n)/S$", gb.total.a, None)]
    assert abs(e.total.a - 1) < 1e-12 and e.total.width < 1e-12
    assert abs(g.total.a - (1 + 3 / 14)) < 1e-12 and abs(p.total.a - (1 - 2 / 14)) < 1e-12
    fig, ax = plt.subplots(figsize=(FULL - 0.35, 1.75))
    rec = {}
    for i, (lab, v, extra) in enumerate(vals):
        y = len(vals) - 1 - i
        if v is None:
            ax.plot(list(extra), [y, y], color=BLUE, lw=2.4, solid_capstyle="butt")
            for xx in extra:
                ax.plot([xx, xx], [y - 0.15, y + 0.15], color=BLUE, lw=0.9)
            ax.text(extra[1] + 0.02, y, "$[%.3f, %.3f]$" % tuple(extra), va="center", fontsize=7)
            rec[lab] = r(list(extra), 3)
        else:
            ax.plot(v, y, marker="o" if i == 0 else "D", ms=5, color=INK, mfc=INK if i == 0 else "white", ls="none")
            ax.text(v + 0.03, y, "%.3f" % v, va="center", fontsize=7)
            rec[lab] = r(v, 3)
    ax.axvline(1, color=MUTED, lw=0.6)
    ax.text(1.0, len(vals) - 0.35, "every probability on $\\{x, \\neg x\\}$", fontsize=6.5, color=INK2, ha="center")
    ax.set_yticks(range(len(vals))); ax.set_yticklabels([x[0] for x in vals][::-1], fontsize=7.4); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.4, 1.65); ax.set_ylim(-0.5, len(vals) - 0.1); ax.set_xlabel("$p(x) + p(\\neg x)$")
    ax.spines["left"].set_visible(False); grid(ax, "x")
    save(fig, "Fig_13_1", rec)


# ---------------------------------------------------------------- the acetaminophen case
def fig_16_3():
    cum = [x for x in csv.DictReader(open(CASE + "/results/cumulative.csv", encoding="utf-8")) if x["variant"] == "main"]
    years = [x["year"] for x in cum]
    parts = [("w_I_A", "association without\nidentification $I_A$", FILL2, "////"), ("w_I_U", "undetermined $I_U$", "white", "...."),
             ("w_I_C", "contradiction $I_C$", INK2, ""), ("w_I_G", "prior weight $I_G$", "white", "")]
    # recompute 2026 with rnel (association-only analyses as a further named type)
    last = cum[-1]
    eA = estimate(Reports(t=float(last["n_t"]), f=float(last["n_f"]), c=float(last["n_c"]), v=float(last["n_v"]),
                          extra=(float(last["n_A"]),)), 2.0)
    wt = decompose(eA.x, "type")
    assert abs(eA.x.lo - float(last["lower"])) < 5e-4 and abs(wt["I1"] - float(last["w_I_A"])) < 5e-4
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.55), gridspec_kw=dict(width_ratios=[1, 1.25]))
    xs = np.arange(len(years))
    for i, x in enumerate(cum):
        lo_, hi_ = float(x["lower"]), float(x["upper"])
        a1.plot([i, i], [lo_, hi_], color=INK, lw=2.4, solid_capstyle="butt")
        a1.plot([i - 0.15, i + 0.15], [lo_, lo_], color=INK, lw=0.8); a1.plot([i - 0.15, i + 0.15], [hi_, hi_], color=INK, lw=0.8)
        a1.text(i, -0.07, "%s" % x["n_analyses"], ha="center", va="top", fontsize=6.5, color=INK2)
    a1.text(-0.9, -0.07, "$n$:", ha="left", va="top", fontsize=6.5, color=INK2)
    a1.set_xticks(xs); a1.set_xticklabels(years, fontsize=7); a1.set_ylim(-0.17, 1.03); a1.set_yticks([0, 0.5, 1])
    a1.set_ylabel("range of $p(P)$"); a1.text(0.0, 1.02, "(a)", transform=a1.transAxes, fontsize=8, va="bottom")
    grid(a1)
    bottom = np.zeros(len(cum))
    rec = {y: {} for y in years}
    for key, lab, fc, h in parts:
        v = np.array([float(x[key]) for x in cum])
        a2.bar(xs, v, bottom=bottom, width=0.62, color=fc, edgecolor=INK, lw=0.6, hatch=h, label=lab)
        for y_, vv in zip(years, v):
            rec[y_][key] = vv
        bottom += v
    for i, x in enumerate(cum):
        assert abs(bottom[i] - float(x["width"])) < 2e-3
    a2.set_xticks(xs); a2.set_xticklabels(years, fontsize=7); a2.set_ylim(0, 1.05); a2.set_yticks([0, 0.5, 1])
    a2.set_ylabel("width and its named parts"); a2.text(0.0, 1.02, "(b)", transform=a2.transAxes, fontsize=8, va="bottom")
    a2.legend(loc="upper right", fontsize=6.2, handlelength=1.4, labelspacing=0.3)
    grid(a2)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "Fig_13_3", dict(years=years, ranges={x["year"]: [float(x["lower"]), float(x["upper"])] for x in cum},
                                parts=rec, recomputed_2026_with_rnel=dict(lower=r(eA.x.lo), upper=r(eA.x.hi), I_A=r(wt["I1"]))))


def fig_16_4():
    a4 = list(csv.DictReader(open(CASE + "/results/a4_classical_cumulative.csv", encoding="utf-8")))
    summ = json.load(open(CASE + "/results/summary.json", encoding="utf-8"))
    idn = summ["identification_estimates_not_pooled"]
    design = {"P06b": "sibling comparison", "P09a": "sibling comparison", "P09b": "negative-control exposure",
              "P10b": "active comparator", "P10c": "negative-control exposure (paternal)"}
    rows = []
    for x in a4:
        rows.append(("naive cumulative RE, %s ($k=%s$)" % (x["year"], x["k"]), float(x["pooled_ratio"]),
                     float(x["ci_low"]), float(x["ci_high"]), "cum"))
    for x in idn:
        lo_, hi_ = (float(v) for v in x["ci"].split("-"))
        kind = "nc" if "negative" in design[x["row_id"]] else "id"
        rows.append(("%s, %d" % (design[x["row_id"]], x["year"]), x["ratio"], lo_, hi_, kind))
    assert abs(rows[-1 - len(idn)][1] - 1.114) < 1e-9          # 2026 naive pooled ratio quoted in Section 12.7.7
    fig, ax = plt.subplots(figsize=(FULL - 1.55, 2.85))
    n = len(rows)
    for i, (lab, m, lo_, hi_, kind) in enumerate(rows):
        y = n - 1 - i + (0.5 if kind == "cum" else 0)
        col = INK if kind == "cum" else BLUE
        ax.plot([lo_, hi_], [y, y], color=col, lw=1.0)
        mk = {"cum": "D", "id": "o", "nc": "o"}[kind]
        ax.plot(m, y, marker=mk, ms=4.5, color=col, mfc="white" if kind == "nc" else col, ls="none")
        ax.text(1.95, y, "%.2f [%.2f, %.2f]" % (m, lo_, hi_), va="center", fontsize=6.6)
    ax.axvline(1, color=MUTED, lw=0.6)
    ax.set_xscale("log"); ax.set_xlim(0.7, 1.9)
    ax.set_xticks([0.75, 1, 1.25, 1.5, 1.75]); ax.set_xticklabels(["0.75", "1", "1.25", "1.5", "1.75"])
    ax.minorticks_off()
    ys = [n - 1 - i + (0.5 if r_[4] == "cum" else 0) for i, r_ in enumerate(rows)]
    ax.set_yticks(ys); ax.set_yticklabels([r_[0] for r_ in rows], fontsize=7); ax.tick_params(axis="y", length=0)
    ax.axhline(len(idn) - 0.25, color=GRID, lw=0.6)
    ax.set_xlabel("ratio (log scale), 95 % interval")
    ax.text(1.95, n + 0.15, "ratio [95 % CI]", fontsize=6.6, va="bottom")
    ax.spines["left"].set_visible(False)
    fig.legend([Line2D([], [], marker="D", color=INK, ls="none"), Line2D([], [], marker="o", color=BLUE, ls="none"),
                Line2D([], [], marker="o", color=BLUE, mfc="white", ls="none")],
               ["population models, pooled (illustration only)", "identification design", "negative control (association of the control)"],
               loc="lower center", bbox_to_anchor=(0.45, 0.0), ncol=3, fontsize=6.5, handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "Fig_13_4", dict(rows=[dict(label=a, ratio=b, ci=[c, d], kind=k) for a, b, c, d, k in rows],
                                sources=["results/a4_classical_cumulative.csv", "results/summary.json"]))


# ---------------------------------------------------------------- the simulation with known truth
def load_sim():
    sim = list(csv.DictReader(open(CASE + "/simulation/results/sim_summary.csv", encoding="utf-8")))
    tab = {}
    for x in sim:
        tab.setdefault(float(x["pi_conf"]), {})[x["policy"]] = (float(x["O1_abs_error"]), float(x["O1_mcse"]))
    return tab


POL = [("typed (ties->IDENT)", "typed rule", INK, "-", "o", 1.6), ("interval-learned", "interval-learned", BLUE, "--", "s", 1.2),
       ("always MORE", "always MORE", MUTED, ":", "^", 0.9), ("always LARGER", "always LARGER", MUTED, "-.", "v", 0.9),
       ("always IDENT", "always IDENT", MUTED, (0, (5, 1.5, 1, 1.5, 1, 1.5)), "D", 0.9)]


def fig_16_5():
    tab = load_sim()
    pis = sorted(tab)
    fig, ax = plt.subplots(figsize=(FULL, 2.6))
    rec = {}
    for key, lab, col, ls, mk, lw in POL:
        y = [tab[p][key][0] for p in pis]; e = [tab[p][key][1] for p in pis]
        ax.errorbar(pis, y, yerr=[1.96 * v for v in e], color=col, ls=ls, marker=mk, ms=4.2, lw=lw,
                    mfc="white" if key != "typed (ties->IDENT)" else col, capsize=0, elinewidth=0.6, label=lab)
        rec[lab] = dict(zip([str(p) for p in pis], y))
    assert rec["typed rule"]["0.5"] == 0.0971 and rec["interval-learned"]["0.0"] == 0.0424 and rec["always IDENT"]["1.0"] == 0.114
    ax.set_xlabel("share of confounded worlds $\\pi$"); ax.set_ylabel("mean absolute error (test half)")
    ax.set_xticks(pis); ax.set_xlim(-0.04, 1.04); grid(ax)
    ax.legend(loc="upper left", fontsize=7, ncol=2, handlelength=3.2)
    save(fig, "Fig_13_5", dict(mean_abs_error=rec, error_bars="1.96 x Monte Carlo SE"))


def fig_16_6():
    tab = load_sim()
    txt = open(CASE + "/simulation/results/posthoc_regret.txt", encoding="utf-8").read()
    block = txt.split("O2_verdict_loss")[0]
    quoted = {m.group(1).strip(): float(m.group(2)) for m in re.finditer(r"^\s+(.+?)\s+max ([0-9.]+)", block, re.M)}
    policies = [k for k in tab[0.0] if k != "oracle"]
    regret = {}
    for k in policies:
        regret[k] = max(tab[p][k][0] - min(tab[p][q][0] for q in policies) for p in tab)
    for k, v in quoted.items():
        assert abs(regret[k] - v) < 1e-9, (k, regret[k], v)
    order = sorted(policies, key=lambda k: regret[k])
    lab = {"typed (ties->IDENT)": "typed rule (ties to IDENT)", "typed (ties->MORE)": "typed rule (ties to MORE)"}
    fig, ax = plt.subplots(figsize=(FULL, 2.3))
    for i, k in enumerate(order):
        y = len(order) - 1 - i
        emph = k in ("typed (ties->IDENT)", "interval-learned")
        ax.barh(y, regret[k], height=0.58, color=INK if k == "typed (ties->IDENT)" else (BLUE if k == "interval-learned" else FILL1),
                edgecolor=INK, lw=0.5, hatch="" if emph else "")
        ax.text(regret[k] + 0.0012, y, "%.4f" % regret[k], va="center", fontsize=7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([lab.get(k, k) for k in order][::-1], fontsize=7.4)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("largest regret over the five scenarios (mean absolute error)")
    ax.set_xlim(0, 0.092); grid(ax, "x"); ax.spines["left"].set_visible(False)
    save(fig, "Fig_13_6", dict(max_regret={k: r(v) for k, v in regret.items()}, checked_against="posthoc_regret.txt",
                                note="regret = error of the policy minus the best non-oracle policy of the same scenario"))


# ---------------------------------------------------------------- Fig 17.1 (diagram)
def fig_17_1():
    XL, XR, BW17, h = 1.45, 6.05, 2.7, 0.95
    B = {  # key: (x, y, text, emphasis)
        "LNP": (XL, 3.6, "$L_{NP}$: linear inequalities\nin $T$, $I$, $F$", True),
        "LQU": (XR, 3.6, "$L^{QU}$: linear inequalities in upper\nprobabilities (Halpern–Pucella)", False),
        "CS": (XL, 1.75, "coherent normalised dual\nneutrosophic probabilities", True),
        "UP": (XR, 1.75, "measurable upper\nprobability structures", False),
        "LS": (XL, 0.0, "$L_S$ on the liftings $S_{\\min}$,\n$S_{\\mathrm{Bel}}$, $S_{\\mathrm{dis}}$ (glut case)", True),
        "SG": (XR, 0.0, "signed liftings\n(off values, charges)", True),
    }
    fig, ax = plt.subplots(figsize=(FULL, 3.2))
    for k, (x, y, t, em) in B.items():
        ax.add_patch(FancyBboxPatch((x - BW17 / 2, y - h / 2), BW17, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=FILL2 if em else "white", ec=INK, lw=0.8, zorder=2))
        ax.text(x, y, t, ha="center", va="center", fontsize=7.2, zorder=3, linespacing=1.15)

    def arrow(p0, p1, two=False):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<|-|>" if two else "-|>", lw=0.8, color=INK,
                                                          mutation_scale=8, shrinkA=0, shrinkB=0), zorder=1)
    a, b = XL + BW17 / 2, XR - BW17 / 2
    arrow((a, 3.78), (b, 3.78)); ax.text((a + b) / 2, 3.85, "$\\tau$", ha="center", va="bottom", fontsize=7.5, color=INK2)
    arrow((b, 3.42), (a, 3.42)); ax.text((a + b) / 2, 3.35, "$\\sigma$", ha="center", va="top", fontsize=7.5, color=INK2)
    arrow((XL, 3.6 - h / 2), (XL, 1.75 + h / 2), two=True)
    ax.text(XL - 0.08, 2.675, "semantics\n(Prop. 12.6.1)", ha="right", va="center", fontsize=6.6, color=INK2)
    arrow((XR, 3.6 - h / 2), (XR, 1.75 + h / 2), two=True)
    ax.text(XR + 0.08, 2.675, "Fact A: sound and\ncomplete (Halpern–\nPucella Thm 4.2)", ha="left", va="center", fontsize=6.6, color=INK2)
    arrow((a, 1.75), (b, 1.75), two=True)
    ax.text((a + b) / 2, 1.82, "Thm 1", ha="center", va="bottom", fontsize=6.6, color=INK2)
    ax.text((a + b) / 2, 2.55, "Thms 12.6.1, 12.6.2:\n$AX_{NP}$ sound and\ncomplete; NP-complete",
            ha="center", va="center", fontsize=6.4, bbox=dict(fc="white", ec=INK2, lw=0.5, pad=1.5), zorder=4)
    arrow((XL, 1.75 - h / 2), (XL, 0.0 + h / 2))
    ax.text(XL - 0.08, 0.875, "Thm 8:\nlifting", ha="right", va="center", fontsize=6.6, color=INK2)
    ax.text(XL, -0.62, "Thm 12.6.4: $AX_S$ complete (Farkas);\nProp. 12.6.3: $T$-consequence = LP, FDE, K3",
            ha="center", va="top", fontsize=6.5, color=INK2)
    ax.text(XR, -0.62, "Thm 12.6.6: $AX_S^{\\pm}$ complete;\nno betting reading", ha="center", va="top", fontsize=6.5, color=INK2)
    ax.set_xlim(-0.3, 8.1); ax.set_ylim(-1.3, 4.25); ax.axis("off")
    save(fig, "Fig_15_1", dict(nodes={k: v[2].replace("\n", " ") for k, v in B.items()},
                                results=["Prop 12.6.1", "Lemmas 12.6.1-12.6.3", "Thm 12.6.1", "Thm 12.6.2", "Thm 12.6.3 (weak logic, not drawn)",
                                         "Thm 12.6.4", "Prop 12.6.3", "Thm 12.6.6", "Halpern & Pucella 2002 Thm 4.2 (Fact A)"]))



if __name__ == "__main__":
    fig_15_1(); fig_15_2(); fig_16_1(); fig_16_2(); fig_16_3(); fig_16_4(); fig_16_5(); fig_16_6(); fig_17_1()
