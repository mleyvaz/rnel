"""The generality thread of Part VI, preface_v2 and ch01_v2: inclusions checked on examples, and worked
examples in which a restricted framework must identify two different evidence states that the neutrosophic
tuple keeps apart. Every number quoted for the generality thread comes from this script.

Sections
  GA  probability is the special case I = 0, T + F = 1 (and the large-sample limit of the evidence tuple)
  GB  Subjective Logic is the retract T + I + F = 1 (IJFS Def. 4.2, Thm 4.3): one SL opinion, three triples
  GC  Belnap-Dunn values are the crisp corners; the Belnap frame is faithful only on T+I<=1, F+I<=1 (Thm 8(d))
  GD  the imprecise Dirichlet model is recovered exactly (Cor. 2): I = U - L on the normalised plane (Thm 1)
  GE  contradiction, undetermined and ill-posed reports: same SL opinion, same credal interval, same vacuity,
      three different neutrosophic tuples and three different actions
  GF  between-source conflict: dispute vs balanced single source (SL, pooled IDM interval, C*)
  GG  refined indeterminacy by source (rnel.neutro_stats.estimate_sources)
  GH  independent components: sums up to 3 are represented on the minimal lifting (Thm 8(a))
  GI  off values: an over-retraction leaves the Beta region (IJFS Thm 10.3)
  GJ  recorded experiment outputs used by the generality thread (exp37, exp38, exp40, exp40c)
Output: generality.json (current directory)   Run: python generality.py
Recorded outputs (GJ) are read from the CACHED summaries in ../../data/cached/experiments.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.getcwd()  # book edition: outputs go to the current working directory
CACHED = os.path.join(HERE, "..", "..", "data", "cached")  # book edition: CACHED experiment summaries
EXP = os.path.join(CACHED, "experiments")
PAPERS = os.path.join(CACHED, "papers")
sys.stdout.reconfigure(encoding="utf-8")

from rnel import Reports, rnel_tuple, fused_contradiction, sl_opinion  # noqa: E402
from rnel.sl import degree_of_conflict  # noqa: E402
from rnel.decide import Policy  # noqa: E402
from rnel import conflict as cf  # noqa: E402
from rnel import neutro_credal as nc  # noqa: E402
from rnel import neutro_stats as ns  # noqa: E402
from rnel.off import retraction_nu, EvidenceLedger, off_from_evidence  # noqa: E402

G = {}


def put(k, v):
    G[k] = v if isinstance(v, (str, bool)) else float(v)


def close(a, b, tol=1e-12):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


W = 2.0

# ---------------------------------------------------------------- GA probability
p = 0.7
put("GA.lift_represents", nc.to_glut_frame((p, 0.0, 1 - p)).represents())
b, d, u, _ = retraction_nu(p, 0.0, 1 - p)
put("GA.retraction_b", b); put("GA.retraction_u", u)
assert close((b, d, u), (p, 1 - p, 0.0))
t30 = rnel_tuple(Reports(t=30, f=10), W)
put("GA.t30f10.T", t30.T); put("GA.t30f10.F", t30.F); put("GA.t30f10.G", t30.G)
big = rnel_tuple(Reports(t=30000, f=10000), W)
put("GA.big.T", big.T); put("GA.big.G", big.G)

# ---------------------------------------------------------------- GB Subjective Logic as a retract
trip = {"complete": (0.4, 0.2, 0.4), "incomplete": (0.4, 0.0, 0.4), "overcomplete": (0.6, 0.3, 0.6)}
ops = {}
for k, x in trip.items():
    b, d, u, _ = retraction_nu(*x)
    ops[k] = (b, d, u)
    put("GB.%s.sum" % k, sum(x))
    put("GB.%s.b" % k, b); put("GB.%s.d" % k, d); put("GB.%s.u" % k, u)
    put("GB.%s.incompleteness" % k, max(0.0, 1 - sum(x)))
    put("GB.%s.paraconsistency" % k, max(0.0, sum(x) - 1))
    put("GB.%s.minimal_lift" % k, nc.to_glut_frame(x).represents())
    put("GB.%s.glut_lower" % k, nc.to_glut_frame(x).glut_lower())
assert close(ops["complete"], ops["incomplete"]) and close(ops["complete"], ops["overcomplete"])
put("GB.same_SL_opinion", True)
put("GB.fixed_point", close(retraction_nu(*trip["complete"])[:3], (0.4, 0.4, 0.2)))

# ---------------------------------------------------------------- GC Belnap-Dunn corners
corners = {"t": (1, 0, 0), "f": (0, 0, 1), "b": (1, 0, 1), "n": (0, 1, 0)}
for k, x in corners.items():
    put("GC.%s.belnap_represents" % k, nc.to_glut_frame(x, "belnap").represents())
    put("GC.%s.minimal_represents" % k, nc.to_glut_frame(x, "minimal").represents())
xg = (0.8, 0.3, 0.7)
put("GC.x.belnap_represents", nc.to_glut_frame(xg, "belnap").represents())
put("GC.x.minimal_represents", nc.to_glut_frame(xg, "minimal").represents())
put("GC.x.glut_lower", nc.to_glut_frame(xg).glut_lower())
env = nc.to_glut_frame(xg, "belnap").envelope()
put("GC.x.belnap_empty", env is None)
if env is not None:
    put("GC.x.belnap_env_T", env[0]); put("GC.x.belnap_env_I", env[1]); put("GC.x.belnap_env_F", env[2])
put("GC.x.disjoint_represents", nc.to_glut_frame(xg, "disjoint").represents())
x_bd = (0.6, 0.3, 0.5)   # T + F > 1 but T + I <= 1, F + I <= 1: inside the Belnap region
put("GC.y.belnap_represents", nc.to_glut_frame(x_bd, "belnap").represents())
put("GC.y.disjoint_represents", nc.to_glut_frame(x_bd, "disjoint").represents())

# ---------------------------------------------------------------- GD IDM and Theorem 1
idm = nc.idm_triple([6, 2], s=2.0)
T, I, F = idm[0]
put("GD.T", T); put("GD.I", I); put("GD.F", F)
put("GD.L", T); put("GD.U", 1 - F); put("GD.width_minus_I", (1 - F - T) - I)
assert abs((1 - F - T) - I) < 1e-12
o = sl_opinion(Reports(t=6, f=2), W)
put("GD.sl_equals_idm", close((o.b, o.u, o.d), (T, I, F)))

# ---------------------------------------------------------------- GE contradiction vs undetermined vs ill-posed
cases = {"C": Reports(t=2, f=2, c=4), "U": Reports(t=2, f=2, v=4), "N": Reports(t=2, f=2, n=4)}
sl_set, ci_set = set(), set()
for k, r in cases.items():
    tp = rnel_tuple(r, W)
    o = tp.to_sl()
    sl_set.add((round(o.b, 12), round(o.d, 12), round(o.u, 12)))
    lo, hi = ns.credal_interval(r, W)
    ci_set.add((round(lo, 12), round(hi, 12)))
    put("GE.%s.sl_b" % k, o.b); put("GE.%s.sl_d" % k, o.d); put("GE.%s.sl_u" % k, o.u)
    put("GE.%s.ci_lo" % k, lo); put("GE.%s.ci_hi" % k, hi)
    for comp in ("T", "F", "C", "U", "N", "G"):
        put("GE.%s.%s" % (k, comp), getattr(tp, comp))
    put("GE.%s.glut_total" % k, ns.estimate(r, W, glut=True, gap=True).total.at())
    put("GE.%s.decision" % k, Policy().decide(tp)[0])
put("GE.n_distinct_SL", len(sl_set)); put("GE.n_distinct_credal", len(ci_set))
assert len(sl_set) == 1 and len(ci_set) == 1

# ---------------------------------------------------------------- GF between-source conflict
dispute, coin = [Reports(t=8), Reports(f=8)], [Reports(t=8, f=8)]
pool = Reports(t=8, f=8)
lo, hi = ns.credal_interval(pool, W)
put("GF.pooled_ci_lo", lo); put("GF.pooled_ci_hi", hi)
o1, o2 = sl_opinion(Reports(t=8), W), sl_opinion(Reports(f=8), W)
put("GF.DC", degree_of_conflict(o1, o2))
put("GF.Cstar_dispute", cf.ConflictState.of_profile(dispute).c_star(W))
put("GF.Cstar_coin", cf.ConflictState.of_profile(coin).c_star(W))
put("GF.Kw_coin", cf.ConflictState.of_profile(coin).Kw)
put("GF.C_def84_dispute", fused_contradiction(dispute, W).C)
put("GF.C_def84_coin", fused_contradiction(coin, W).C)
# credal set over the sources (one IDM interval per source): keeps the dispute too, at V x 2 numbers
put("GF.src1_ci_lo", ns.credal_interval(Reports(t=8), W)[0]); put("GF.src2_ci_hi", ns.credal_interval(Reports(f=8), W)[1])

# ---------------------------------------------------------------- GG refined indeterminacy by source
src_a = {"lab A": Reports(t=4, c=3), "lab B": Reports(f=2, n=1)}
src_b = {"lab A": Reports(t=4, n=1), "lab B": Reports(f=2, c=3)}
for name, srcs in (("a", src_a), ("b", src_b)):
    e = ns.estimate_sources(srcs, W)
    lo, hi = e.x.range()
    put("GG.%s.lo" % name, lo); put("GG.%s.hi" % name, hi)
    for key, v in ns.decompose(e.x, by="source").items():
        put("GG.%s.by_source.%s" % (name, key if isinstance(key, str) else str(key)), v)
    for key, v in ns.decompose(e.x, by="type").items():
        put("GG.%s.by_type.%s" % (name, key if isinstance(key, str) else str(key)), v)
assert abs(G["GG.a.lo"] - G["GG.b.lo"]) < 1e-12 and abs(G["GG.a.hi"] - G["GG.b.hi"]) < 1e-12

# ---------------------------------------------------------------- GH independent components (sum up to 3)
for name, x in (("x", (0.9, 0.8, 0.7)), ("ones", (1.0, 1.0, 1.0))):
    gl = nc.to_glut_frame(x)
    put("GH.%s.sum" % name, sum(x)); put("GH.%s.represents" % name, gl.represents())
    put("GH.%s.glut_lower" % name, gl.glut_lower())
    put("GH.%s.belnap_represents" % name, nc.to_glut_frame(x, "belnap").represents())
    b, d, u, _ = retraction_nu(*x)
    put("GH.%s.sl_b" % name, b); put("GH.%s.sl_d" % name, d); put("GH.%s.sl_u" % name, u)
put("GH.x.sure_loss_degree", nc.sure_loss_degree((0.9, 0.8, 0.7)))
bp = retraction_nu(0.45, 0.4, 0.35)[:3]
put("GH.prop.same_SL_as_x", close(bp, retraction_nu(0.9, 0.8, 0.7)[:3])); put("GH.prop.sum", 1.2)

# ---------------------------------------------------------------- GI off values
L = EvidenceLedger().report("trialA", for_x=4).report("cohortB", against_x=2).report("cohortC", for_x=3)
L.retract_source("trialA")
oa = L.opinion()
put("GI.after_b", oa.b); put("GI.after_d", oa.d); put("GI.after_u", oa.u)
off = off_from_evidence(3 - 4, 2)
put("GI.anti_b", off.b); put("GI.anti_d", off.d); put("GI.anti_u", off.u)
put("GI.anti_beta_ok", bool(off.beta_admissible))

# ---------------------------------------------------------------- GK dissonance = C* + within-source share
def diss(profile, W=2.0):
    R = sum(r for r, _ in profile); S = sum(s_ for _, s_ in profile)
    return 2 * min(R, S) / (R + S + W)
for name, prof in (("dispute", [(8, 0), (0, 8)]), ("coin", [(8, 8)]), ("abc", [(6, 0), (0, 6), (3, 3)])):
    st = cf.ConflictState.of_profile(prof)
    put("GK.%s.dissonance" % name, diss(prof)); put("GK.%s.Cstar" % name, st.c_star(W))
    put("GK.%s.Cwithin" % name, st.c_within(W))
    assert abs(diss(prof) - st.c_star(W) - st.c_within(W)) < 1e-12
put("GK.identity_holds", True)

# ---------------------------------------------------------------- GL Example 18.1 (two views, ten neighbours each)
put("GL.C_split_views", fused_contradiction([Reports(t=10), Reports(f=10)], W).C)
put("GL.C_balanced_views", fused_contradiction([Reports(t=5, f=5), Reports(t=5, f=5)], W).C)
put("GL.same_pooled", True)

# ---------------------------------------------------------------- GT guided tour of Chapter 1 (t=2, f=2, c=4)
rt = Reports(t=2, f=2, c=4)
tt = rnel_tuple(rt, W)
trip_norm = (tt.T, 1 - tt.T - tt.F, tt.F)                      # normalised triple of the evidence (T, I, F)
put("GT.norm.T", trip_norm[0]); put("GT.norm.I", trip_norm[1]); put("GT.norm.F", trip_norm[2])
put("GT.norm.coherent_lift", nc.to_glut_frame(trip_norm).represents())
est = ns.estimate(rt, W)
put("GT.est.lo", est.x.range()[0]); put("GT.est.hi", est.x.range()[1])
for key, v in ns.decompose(est.x, by="type").items():
    put("GT.est.by_type.%s" % key, v)
Sg = 2 + 2 + 4 + W
glut = ((2 + 4) / Sg, W / Sg, (2 + 4) / Sg)                    # glut reading: contradictory reports count for both
put("GT.glut.T", glut[0]); put("GT.glut.I", glut[1]); put("GT.glut.F", glut[2])
put("GT.glut.belnap", nc.to_glut_frame(glut, "belnap").represents())
put("GT.glut.minimal", nc.to_glut_frame(glut).represents())
put("GT.glut.glut_lower", nc.to_glut_frame(glut).glut_lower())
put("GT.glut.sure_loss", nc.sure_loss_degree(glut))
put("GT.relfreq", 2 / 4)

# ---------------------------------------------------------------- GJ recorded outputs
J = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731
s40 = J(EXP + "/40_monotonicity_conflict/results/summary.json")
for k in ("k=4", "k=6", "k=10"):
    if k in s40:
        a = s40[k]["auroc_final"]
        put("GJ.e40.%s.auroc_u" % k, a["u_rnel"]["S2_vs_S1S3"])
        put("GJ.e40.%s.auroc_Cstar" % k, a["C_star"]["S2_vs_S1S3"])
        put("GJ.e40.%s.auroc_kappa" % k, a["kappa"]["S2_vs_S1S3"])
        put("GJ.e40.%s.u_monotone_share" % k, s40[k]["H1_u_rnel_monotone_share"])
        fm = s40[k]["final_means"]
        for sc in fm:
            put("GJ.e40.%s.%s.Cstar" % (k, sc), fm[sc]["C_star"])
            put("GJ.e40.%s.%s.kappa" % (k, sc), fm[sc]["kappa"])
put("GJ.e40.dempster_kappa", s40["worked_example"]["kappa"])
put("GJ.e40.dempster_mOmega", s40["worked_example"]["m_Omega_after_dempster"])
s40c = J(EXP + "/40c_margin_normaliser/results/summary.json")
for m in ("C_star", "kappa", "u"):
    for cond, v in s40c["testB"][m]["bal_acc"].items():
        put("GJ.e40c.%s.%s" % (m, cond), v)
s37 = J(EXP + "/37_neutro_vs_credal_economy/results/summary.json")
m37 = s37["memory_K20_V10000"]
put("GJ.e37.mem_R1_bytes", m37["R1"]); put("GJ.e37.mem_R2_bytes", m37["R2"])
put("GJ.e37.mem_C1_bytes", m37["C1_vertices"]); put("GJ.e37.ratio", s37["ratio_C1vert_over_R1_K20_V10000"])
put("GJ.e37.max_err_int", s37["max_err_int"]); put("GJ.e37.max_err_real", s37["max_err_real"])
put("GJ.e37.us_R1_K20", s37["K20"]["online_us_V10000"]["R1"]); put("GJ.e37.us_C1_K20", s37["K20"]["online_us_V10000"]["C1"])
s38 = J(EXP + "/38_conflict_aware_streaming_fusion/results/localization_full.json")
for sc in ("P_rho0.3", "P_all", "I_all"):
    for m, v in s38["medians"][sc].items():
        put("GJ.e38.median.%s.%s" % (sc, m), v)
c38 = s38["counts_vs_rnel_stream"]["P_rho0.3"]["vote_stream"]
put("GJ.e38.vote_ties_or_beats_P03", c38["0_ties_or_beats"]); put("GJ.e38.cells_P03", c38["n_cells"])
put("GJ.e38.max_P_all.rnel_stream", s38["maxima"]["P_all"]["rnel_stream"])
h38 = J(EXP + "/38_conflict_aware_streaming_fusion/results/hypotheses.json")["dsa"]["rnel_stream_vs_mean|P|0.1"][0]
assert h38["cond"] == "drift_P_0.1"
put("GJ.e38.dsa_drift.diff", h38["diff"]); put("GJ.e38.dsa_drift.lo", h38["ci"][0]); put("GJ.e38.dsa_drift.hi", h38["ci"][1])

json.dump(G, open(os.path.join(OUT, "generality.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("generality:", len(G))
for k in sorted(G):
    v = G[k]
    print("%-48s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))
