"""Every number quoted in the evidence-reframe material (preface_v2, ch01_introduction_v2, Part VI).

Two sources, kept apart in the output:
  A. worked examples computed here with the public API of the rnel library (imported from PYTHONPATH;
     the branch/commit of the enclosing git checkout, if any, is recorded);
  B. recorded experiment outputs, read from the CACHED summaries in ../../data/cached/ (JSON written by the
     experiment scripts; nothing is recomputed or retyped; see ../../data/cached/README.md).

Output (book edition: current directory): numbers.json {key: value}, plus numbers_sources.json {key: source}.
Run:    python compute_numbers.py   (renamed 3-oct: a file called numbers.py shadowed the stdlib module `numbers`
        imported by torch, so the torch subprocess re-ran this script recursively and exhausted memory)
"""
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.getcwd()  # book edition: outputs go to the current working directory
CACHED = os.path.join(HERE, "..", "..", "data", "cached")  # book edition: CACHED experiment summaries
EXP = os.path.join(CACHED, "experiments")
PAPERS = os.path.join(CACHED, "papers")
sys.stdout.reconfigure(encoding="utf-8")
import rnel  # noqa: E402
RNEL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(rnel.__file__))))

from rnel import Reports, rnel_tuple, fused_contradiction, sl_opinion, Opinion  # noqa: E402
from rnel.sl import degree_of_conflict, cumulative_fusion  # noqa: E402
from rnel.decide import Policy  # noqa: E402
from rnel import conflict as cf  # noqa: E402
from rnel import operators as ops  # noqa: E402
from rnel.off import EvidenceLedger, off_from_evidence  # noqa: E402

N, SRC = {}, {}


def put(key, value, source):
    N[key] = float(value)
    SRC[key] = source


def J(path):
    return json.load(open(path, encoding="utf-8"))


# ======================================================================== A. worked examples (rnel)
A = "rnel worked example (compute_numbers.py, section A)"
W = 2.0

# WE1: two confident outlets that disagree vs one balanced source (Chapter VI.1, Example 1)
oa, ob = sl_opinion(Reports(t=8)), sl_opinion(Reports(f=8))
put("we1.bA", oa.b, A); put("we1.uA", oa.u, A); put("we1.PA", oa.projected, A); put("we1.PB", ob.projected, A)
put("we1.DC", degree_of_conflict(oa, ob), A)
fu = cumulative_fusion(oa, ob)
put("we1.fused_b", fu.b, A); put("we1.fused_d", fu.d, A); put("we1.fused_u", fu.u, A)
coin = sl_opinion(Reports(t=8, f=8))
assert abs(coin.b - fu.b) < 1e-12 and abs(coin.u - fu.u) < 1e-12
x = fused_contradiction([Reports(t=8), Reports(f=8)])
put("we1.C_def84", x.C, A); put("we1.T", x.T, A); put("we1.G", x.G, A)
put("we1.coin_C", fused_contradiction([Reports(t=8, f=8)]).C, A)
put("we1.total_def84", x.total, A)
st = cf.ConflictState.of_profile([Reports(t=8), Reports(f=8)])
put("we1.Cstar", st.c_star(W), A); put("we1.Kb", st.Kb, A)
put("we1.coin_Cstar", cf.ConflictState.of_profile([Reports(t=8, f=8)]).c_star(W), A)
put("we1.coin_Kw", cf.ConflictState.of_profile([Reports(t=8, f=8)]).Kw, A)
d1 = Policy().decide(x)[0]
d2 = Policy().decide(fused_contradiction([Reports(t=8, f=8)]))[0]
N["we1.decision_dispute"], N["we1.decision_coin"] = d1, d2
SRC["we1.decision_dispute"] = SRC["we1.decision_coin"] = A

# WE2: the evidence tuple of one claim (Chapter VI.1, Example 2)
r2 = Reports(t=4, f=2, c=1, v=1, n=2)
t2 = rnel_tuple(r2)
for k, v in t2.as_dict().items():
    put("we2." + k, v, A)
put("we2.Sigma", 4 + 2 + 1 + 1 + 2 + W, A)
put("we2.total", t2.total, A)
o2 = t2.to_sl()
put("we2.sl_b", o2.b, A); put("we2.sl_d", o2.d, A); put("we2.sl_u", o2.u, A); put("we2.sl_P", o2.projected, A)
N["we2.decision"] = Policy().decide(t2)[0]; SRC["we2.decision"] = A

# WE3: order dependence of the sequential pairwise rule vs the order-free C* (Chapter VI.1, Example 3)
prof = [(6, 0), (0, 6), (3, 3)]
seq = {"".join("ABC"[i] for i in p): cf.def84_sequential([prof[i] for i in p]) for p in itertools.permutations(range(3))}
for k, v in seq.items():
    put("we3.seq_" + k, v, A)
put("we3.seq_min", min(seq.values()), A); put("we3.seq_max", max(seq.values()), A)
put("we3.maxpair", cf.def84_maxpair(prof), A)
cs = {"".join("ABC"[i] for i in p): cf.c_star([prof[i] for i in p]) for p in itertools.permutations(range(3))}
assert max(cs.values()) - min(cs.values()) < 1e-15
put("we3.Cstar", cs["ABC"], A)
st3 = cf.ConflictState.of_profile(prof)
put("we3.R", st3.R, A); put("we3.S", st3.S, A); put("we3.Kw", st3.Kw, A); put("we3.Kb", st3.Kb, A)
put("we3.Cwithin", st3.c_within(W), A)

# WE4: monotone evidence gate of rnel.decide (Chapter VI.1, Section on decisions)
p = Policy()
N["we4.f3"] = p.decide(rnel_tuple(Reports(f=3)))[0]
N["we4.t1f3_new"] = p.decide(rnel_tuple(Reports(t=1, f=3)))[0]
N["we4.t1f3_old"] = Policy(side_gate=False).decide(rnel_tuple(Reports(t=1, f=3)))[0]
for k in ("we4.f3", "we4.t1f3_new", "we4.t1f3_old"):
    SRC[k] = A
put("we4.required_side", p.required_side_evidence(), A)
put("we4.G_f3", rnel_tuple(Reports(f=3)).G, A); put("we4.G_t1f3", rnel_tuple(Reports(t=1, f=3)).G, A)

# WE5: retraction and inversion with the evidence ledger (Chapter VI.1, Section on signed evidence)
L = EvidenceLedger().report("trialA", for_x=4).report("cohortB", against_x=2).report("cohortC", for_x=3)
o_before = L.opinion()
put("we5.before_b", o_before.b, A); put("we5.before_d", o_before.d, A); put("we5.before_u", o_before.u, A)
L.retract_source("trialA")
o_after = L.opinion()
put("we5.after_b", o_after.b, A); put("we5.after_d", o_after.d, A); put("we5.after_u", o_after.u, A)
o_un = off_from_evidence(3 - 4, 2)  # the retracted trial counted once more, as anti-evidence (Def. 10.2 of the IJFS paper)
put("we5.anti_b", o_un.b, A); put("we5.anti_d", o_un.d, A); put("we5.anti_u", o_un.u, A)
N["we5.anti_beta_ok"] = bool(o_un.beta_admissible); SRC["we5.anti_beta_ok"] = A
put("we5.anti_P", o_un.projected, A)

# WE6: operators (priority product), De Morgan and coarsening (Chapter VI.1, Section on operators)
xa = rnel_tuple(Reports(t=6, f=1, c=1)).as_dict()
yb = rnel_tuple(Reports(t=2, f=2, n=2)).as_dict()
z_and = ops.rnel_and(xa, yb)
z_or = ops.rnel_or(xa, yb)
dm = ops.rnel_not(ops.rnel_and(ops.rnel_not(xa), ops.rnel_not(yb)))
put("we6.demorgan_err", max(abs(dm[k] - z_or[k]) for k in z_or), A)
for k in ("T", "F", "C", "N", "G"):
    put("we6.and_" + k, z_and[k], A)
put("we6.and_mass", ops.mass(z_and), A)
put("we6.x_T", xa["T"], A); put("we6.y_T", yb["T"], A)

# WE7: the typed head reduces to evidential deep learning (Chapter VI.2); computed by numbers_torch.py in a
# separate process (torch import order), merged here
sys.stdout.flush()
subprocess.run([sys.executable, os.path.join(HERE, "numbers_torch.py")], check=True, cwd=OUT)
for k, v in J(os.path.join(OUT, "numbers_torch.json")).items():
    put(k, v, "rnel worked example (numbers_torch.py)")

# WE8: the three evidence regimes of a triple on {x, not x} (Chapter 1, Observation 1.1, Figure 1.2)
def regimes(T, I, F):
    s = T + I + F
    if abs(s - 1) < 1e-12:
        m = {"x": T, "notx": F, "Theta": I}
        return "DS", m, dict(Bel=T, Pl=1 - F)
    if s < 1:
        m = {"x": T, "notx": F, "Theta": I, "empty": 1 - s}
        return "TBM", m, dict(Bel=T, Pl=T + I)
    if T + I <= 1 + 1e-12 and F + I <= 1 + 1e-12:
        g = s - 1
        m = {"x": T - g, "notx": F - g, "Theta": I, "glut": g}
        return "DSm", m, dict(Bel=m["x"] + g, Belnot=m["notx"] + g)
    return "lifting", {}, {}


for name, trip in {"ds": (0.5, 0.2, 0.3), "tbm": (0.4, 0.2, 0.2), "dsm": (0.6, 0.2, 0.5), "lift": (0.9, 0.3, 0.6)}.items():
    reg, m, b = regimes(*trip)
    N["we8.%s.regime" % name] = reg; SRC["we8.%s.regime" % name] = A
    for k, v in {**m, **b}.items():
        put("we8.%s.%s" % (name, k), v, A)
    if m:
        assert abs(sum(m.values()) - 1) < 1e-12 and min(m.values()) >= -1e-12

# ======================================================================== B. recorded experiment outputs
def b(key, path, *ks, idx=None):
    v = J(path)
    for k in ks:
        v = v[k]
    if idx is not None:
        v = v[idx] if not isinstance(idx, tuple) else v[idx[0]][idx[1]]
    put(key, v, os.path.relpath(path, EXP if path.startswith(EXP) else PAPERS).replace("\\", "/") + " :: " + "/".join(map(str, ks)) + ("" if idx is None else " [%s]" % (idx,)))


# exp23 (typed head on fact-checking data; nested analysis)
p23 = EXP + "/23_rnel_real/results/summary_nested.json"
for ds in ("averitec", "climate"):
    for m in ("rnel", "edl", "softmax", "sl", "rule", "logreg"):
        b("e23.%s.macro.%s" % (ds, m), p23, ds, "macro_f1", m)
        b("e23.%s.f1C.%s" % (ds, m), p23, ds, "f1_per_class", m, "C")
    for h in ("R1_rnel_minus_edl_macro", "R2_rnel_minus_softmax_F1C", "R2_rnel_minus_rule_F1C", "desc_rnel_minus_sl_F1C"):
        b("e23.%s.%s" % (ds, h), p23, ds, h, idx=0)
        b("e23.%s.%s.lo" % (ds, h), p23, ds, h, idx=(1, 0))
        b("e23.%s.%s.hi" % (ds, h), p23, ds, h, idx=(1, 1))
    b("e23.%s.prev_C" % ds, p23, ds, "prauc_C", "prevalence")
    for m in ("rnel_I_vac_on_rnel_errors", "edl_vac_on_edl_errors", "softmax_1_minus_msp_on_softmax_errors"):
        b("e23.%s.D3.%s" % (ds, m), p23, ds, "D3_error_detection_auroc", m)
    b("e23.%s.n" % ds, p23, ds, "n")
    b("e23.%s.nC" % ds, p23, ds, "class_counts", "C")
b("e23.chaos.rnel_I_amb", p23, "D1_chaos_spearman_with_human_entropy", "rnel_I_amb")
b("e23.chaos.softmax_pU", p23, "D1_chaos_spearman_with_human_entropy", "softmax_pU")
b("e23.chaos.softmax_entropy", p23, "D1_chaos_spearman_with_human_entropy", "softmax_entropy")
b("e23.D4_reduction", p23, "D4_reduction_max_abs_error")
ps = EXP + "/23_rnel_real/paper_section/section_numbers.json"
for ds in ("averitec", "climate"):
    for cls in ("T", "C", "N"):
        for comp in ("C", "N", "G"):
            b("e23.%s.def81.%s.%s" % (ds, cls, comp), ps, ds + "_def81_by_class", cls, comp)

# exp24 (weak supervision: per-source evidence, true/false labels only)
p24 = EXP + "/24_rnel_weak/results/summary.json"
for ds in ("climate", "averitec"):
    for m in ("C_src", "D_fused", "D_DS", "C_src_NLI", "zeroshot_rule", "count", "vacuity_pp"):
        b("e24.%s.auroc.%s" % (ds, m), p24, ds, "auroc", m)
    b("e24.%s.diff_Csrc_Dfused" % ds, p24, ds, "diff_Csrc_minus_Dfused", idx=0)
    b("e24.%s.diff_Csrc_Dfused.lo" % ds, p24, ds, "diff_Csrc_minus_Dfused", idx=(1, 0))
    b("e24.%s.diff_Csrc_Dfused.hi" % ds, p24, ds, "diff_Csrc_minus_Dfused", idx=(1, 1))
    b("e24.%s.diff_Csrc_NLI" % ds, p24, ds, "diff_Csrc_minus_CsrcNLI", idx=0)
    b("e24.%s.N_vacuity_pp" % ds, p24, ds, "N_auroc", "vacuity_pp")
    b("e24.%s.tf_pp" % ds, p24, ds, "tf_macroF1", "pp_mean")
    b("e24.%s.tf_ds" % ds, p24, ds, "tf_macroF1", "ds")

# exp25 (regression, sensors as sources)
p25 = EXP + "/25_rnel_regression/summary.json"
p25i = EXP + "/25_rnel_regression/summary_intel.json"
for m in ("C_RNEL", "MoNIG_var", "Dist", "G_RNEL"):
    b("e25.gas.fault.%s" % m, p25, "val_abs_err", "auroc_fault", m)
    b("e25.gas.bigerr.%s" % m, p25, "val_abs_err", "auroc_bigerr", m)
    b("e25.intel.fault.%s" % m, p25i, "val_abs_err", "auroc_fault", m)
    b("e25.intel.selrmse.%s" % m, p25i, "val_abs_err", "sel_rmse80_TCF", m)

# exp35 (C* read-out on trained multi-view evidential models, RCML protocol)
p35 = EXP + "/35_rnel_head_niche/results/summary.json"
for ds in ("HandWritten", "Scene", "PIE"):
    for m in ("R1_Cstar", "B1_vacuity", "B2_dissonance", "B3_RCML_DC", "B4_TMC_K", "B5_plurality", "R2_Def84_maxpair"):
        b("e35.%s.auroc.%s" % (ds, m), p35, ds, "RCML", "auroc", m, idx=0)
    for m in ("Cstar", "Plurality", "TMC_K", "RCML_DC"):
        b("e35.%s.loc.%s" % (ds, m), p35, ds, "RCML", "loc", m, idx=0)
    b("e35.%s.order_changed" % ds, p35, ds, "RCML", "order", "share_changed", idx=0)
    for mdl in ("RCML", "RNEL"):
        b("e35.%s.acc_clean.%s" % (ds, mdl), p35, ds, mdl, "acc_clean", idx=0)
        b("e35.%s.acc_corrupt.%s" % (ds, mdl), p35, ds, mdl, "acc_corrupt", idx=0)

# exp36 (real conflict: credal width vs C*), with the source-count stratification
for ds in ("averitec", "climate"):
    p36 = EXP + "/36_credal_vs_neutro_real_conflict/results/summary_%s.json" % ds
    for m in ("C*_K", "width_hull", "width_src", "entropy"):
        b("e36.%s.t1.%s" % (ds, m), p36, "task1_conflict", m, "auroc_mean")
        b("e36.%s.t2.%s" % (ds, m), p36, "task2_error", m, "auroc_mean")
    b("e36.%s.n_conflict" % ds, p36, "n_conflict")
    b("e36.%s.n_items" % ds, p36, "n_items")
pconf = PAPERS + "/conformal_conflict/code/results/s_confound_exp36.json"
for m in ("C*_K", "width_src", "width_hull", "entropy"):
    b("e36.averitec.strat.%s" % m, pconf, "averitec", m, "auroc_within_S_strata_mean5")
b("e36.averitec.auroc_S", pconf, "averitec", "auroc_S")

# exp37 (economy of the state) and exp38 (streaming fusion), recorded summaries
p37 = EXP + "/37_neutro_vs_credal_economy/results/summary.json"
p38 = EXP + "/38_conflict_aware_streaming_fusion/results/localization_full.json"
try:
    s37 = J(p37)
    SRC["e37.summary_keys"] = "%s keys: %s" % (os.path.relpath(p37, EXP), ",".join(list(s37)[:12]))
except Exception as ex:  # pragma: no cover
    print("exp37 summary not read:", ex)

# exp40 (synthetic: monotone vacuity, conflict kept), k = 6
p40 = EXP + "/40_monotonicity_conflict/results/summary.json"
for sc in ("S1_agreement", "S2_conflict", "S3_ambiguity"):
    b("e40.%s.Cstar" % sc, p40, "k=6", "final_means", sc, "C_star")
    b("e40.%s.kappa" % sc, p40, "k=6", "final_means", sc, "kappa")
    b("e40.%s.u" % sc, p40, "k=6", "final_means", sc, "u_rnel")

# exp26 v2 and exp27 (typed per-point tuple in multi-view data)
p26 = EXP + "/26_rnel_cluster/summary_v2.json"
for comp in ("C", "U", "N"):
    pass
b("e26.conflict.C", p26, "specificity_test", "conflict", "C")
b("e26.conflict.max_js", p26, "specificity_test", "conflict", "max_js")
b("e26.ambiguous.U", p26, "specificity_test", "ambiguous", "U")
b("e26.ambiguous.view_entropy", p26, "specificity_test", "ambiguous", "view_entropy")
b("e26.outlier.N", p26, "specificity_test", "outlier", "N")
b("e26.outlier.knn_ratio", p26, "specificity_test", "outlier", "knn_ratio")
p27 = EXP + "/27_rnel_cluster_paper/summary_paired.json"
for t in ("conflict", "ambiguous", "outlier"):
    b("e27.dsa.%s.diff" % t, p27, "dsa", t, "diff_vs_best")
    b("e27.dsa.%s.won" % t, p27, "dsa", t, "repetitions_won")
b("e27.dsa.outlier.direct_diff", p27, "dsa", "outlier", "direct_diff")

# exp28 (RNEL-MVC, confirmatory DSA and HAR)
p28 = EXP + "/28_rnel_mvc/summary.json"
for ds in ("dsa", "har"):
    b("e28.%s.H1.diff" % ds, p28, ds, "H1", "diff")
    b("e28.%s.H1.lo" % ds, p28, ds, "H1", "ci", idx=0)
    b("e28.%s.H1.hi" % ds, p28, ds, "H1", "ci", idx=1)
    for sc in ("S1", "S2"):
        b("e28.%s.H2.%s" % (ds, sc), p28, ds, "H2", sc, "diff")
        b("e28.%s.H2.%s.lo" % (ds, sc), p28, ds, "H2", sc, "ci", idx=0)
        b("e28.%s.H2.%s.hi" % (ds, sc), p28, ds, "H2", sc, "ci", idx=1)
        b("e28.%s.H2.%s.wins" % (ds, sc), p28, ds, "H2", sc, "wins")
        b("e28.%s.H3.%s" % (ds, sc), p28, ds, "H3", sc, "diff")
    b("e28.%s.H4.weight" % ds, p28, ds, "H4", "mean_weight")
    b("e28.%s.H4.uniform" % ds, p28, ds, "H4", "uniform")
    b("e28.%s.S4.wmvec_weight" % ds, p28, ds, "table", "S4|wmvec", "weight_bad_view")
    b("e28.%s.S1.rnel_seconds" % ds, p28, ds, "table", "S1|rnel", "seconds")
    b("e28.%s.S1.ecm_seconds" % ds, p28, ds, "table", "S1|ecm", "seconds")
    b("e28.%s.S1.rnel_auroc" % ds, p28, ds, "table", "S1|rnel", "auroc_contam")
    b("e28.%s.S1.rnel_noC_auroc" % ds, p28, ds, "table", "S1|rnel_no_c", "auroc_contam")
    b("e28.%s.S0.nmi.rnel" % ds, p28, ds, "table", "S0|rnel", "nmi_clean")
    b("e28.%s.S0.nmi.ecm" % ds, p28, ds, "table", "S0|ecm", "nmi_clean")

# exp30 (new contaminations, multi-view outlier baselines)
p30 = EXP + "/30_unified/summary.json"
for ds in ("dsa", "har"):
    for sc in ("S8", "S1", "S2", "S5", "S6"):
        v = J(p30)[ds]["cluster"]["vs_best_S%s" % sc[1:]] if False else None
    ma = J(p30)[ds]["cluster"]["mean_auroc"]
    for sc in ("S8", "S1", "S2", "S5", "S6", "S7"):
        b("e30.%s.%s.rnel" % (ds, sc), p30, ds, "cluster", "mean_auroc", sc, "rnel_mvc")
        best = max((v, k) for k, v in ma[sc].items() if k != "rnel_mvc")
        put("e30.%s.%s.best" % (ds, sc), best[0], "%s :: %s/cluster/mean_auroc/%s/%s (best non-RNEL score)"
            % (os.path.relpath(p30, EXP), ds, sc, best[1]))
        N["e30.%s.%s.best_name" % (ds, sc)] = best[1]
        b("e30.%s.%s.knn" % (ds, sc), p30, ds, "cluster", "mean_auroc", sc, "knn_inconsistency")

# exp31 (active learning, leakage-free redesign)
p31 = EXP + "/31_active_redesign/summary.json"
for ds in ("dsa", "har"):
    for sc in ("S1", "S2", "S8"):
        b("e31.%s.AL1.%s" % (ds, sc), p31, ds, "AL1_%s" % sc, "mean")
        b("e31.%s.AL2.%s" % (ds, sc), p31, ds, "AL2_%s" % sc, "mean")
        b("e31.%s.AL3.%s" % (ds, sc), p31, ds, "AL3_%s" % sc, "mean")
        b("e31.%s.cold_vs_kmeans.%s" % (ds, sc), p31, ds, "cold_%s" % sc, "rnel_vs_kmeans_margin", "mean")
    b("e31.%s.l2t.rnel_margin.S1" % ds, p31, ds, "labels_to_target_S1", "rnel|margin", "median_labels")
    b("e31.%s.l2t.rnel_margin_rel.S1" % ds, p31, ds, "labels_to_target_S1", "rnel|margin_rel", "median_labels")
    b("e31.%s.l2t.rnel_margin.S1.reached" % ds, p31, ds, "labels_to_target_S1", "rnel|margin", "reached")
    b("e31.%s.l2t.rnel_margin_rel.S1.reached" % ds, p31, ds, "labels_to_target_S1", "rnel|margin_rel", "reached")

# exp32 (ablation of the conflict factor, harmful vs harmless)
p32 = EXP + "/32_conflict_ablation/summary.json"
for ds in ("dsa", "har"):
    for m in ("full", "support_only", "conflict_only", "no_conflict", "knn_inconsistency"):
        b("e32.%s.S8.%s" % (ds, m), p32, ds + "_S8", m)
p32h = EXP + "/32_conflict_ablation/harmful_vs_harmless_summary.json"
for ds in ("dsa", "har"):
    for m in ("rnel_full", "rnel_conflict_only", "knn_inconsistency", "hoad"):
        b("e32.%s.swapped.%s" % (ds, m), p32h, ds, "auroc_swapped", m)

# provenance of the checked-out rnel
try:
    N["rnel.branch"] = subprocess.check_output(["git", "-C", RNEL, "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    N["rnel.commit"] = subprocess.check_output(["git", "-C", RNEL, "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
except Exception:  # pragma: no cover
    pass

json.dump(N, open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump(SRC, open(os.path.join(OUT, "numbers_sources.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("numbers:", len(N))
for k in sorted(N):
    if k.startswith(("we", "e3", "e2", "e4")):
        v = N[k]
        print("%-45s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))
