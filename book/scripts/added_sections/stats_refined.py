"""(v13) Every number of Section 12.7 (neutrosophic statistics with refined indeterminacy), of the reconciliation
"I versus Walley's gap" and of Section 12.8 is produced or read here.

Sources (read-only):
  - rnel.neutro_stats (imported from PYTHONPATH): the two-source demo and random checks;
  - the acetaminophen/ASD case (CACHED copy in ../../data/cached/paracetamol_case/results/{summary.json,cumulative.csv,
    a4_classical_cumulative.csv,coding.csv});
  - the simulation with known truth (CACHED copy in ../../data/cached/paracetamol_case/simulation/results/
    {sim_summary.csv,posthoc_regret.txt} and simulation/00_PLAN_SIMULACION.md).
Writes stats_refined.json and stats_refined.out to the current directory (book edition).
"""
import sys, os, json, csv, random, hashlib, subprocess, re
sys.stdout.reconfigure(encoding="utf-8")
import rnel
from rnel.neutro_stats import estimate, estimate_sources, decompose, interval_sum, credal_interval
from rnel.tuple import Reports

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.getcwd()  # book edition: outputs go to the current working directory
CASE = os.path.join(HERE, "..", "..", "data", "cached", "paracetamol_case")
RNEL_PKG = os.path.dirname(os.path.abspath(rnel.__file__))
RNEL = os.path.dirname(os.path.dirname(RNEL_PKG))  # repository root when rnel is imported from <repo>/src
out = {}
lines = []


def say(*a):
    s = " ".join(str(x) for x in a); lines.append(s); print(s)


def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()


out["rnel_commit"] = subprocess.run(["git", "-C", RNEL, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
out["md5"] = {os.path.relpath(p, RNEL).replace(os.sep, "/"): md5(p) for p in
              [os.path.join(RNEL_PKG, "neutro_stats.py"), os.path.join(RNEL, "examples", "neutro_stats_demo.py")]
              if os.path.exists(p)}

# ------------------------------------------------------------------ 1. two-source worked example (= examples/neutro_stats_demo.py)
W = 2.0
src = {"trial A": Reports(t=4, c=2), "cohort B": Reports(f=3, c=1, n=2)}
pooled = Reports(t=4, f=3, c=3, n=2)
S = 4 + 3 + 3 + 2 + W
lo, hi = credal_interval(pooled, W)
e = estimate_sources(src, W)
coef = {("%s[%s]" % k if isinstance(k, tuple) else k): v for k, v in e.x.terms.items()}
by_type = decompose(e.x, "type"); by_type_rel = decompose(e.x, "type", relative=True)
by_src = decompose(e.x, "source"); by_src_rel = decompose(e.x, "source", relative=True)
ia = interval_sum(e.x, e.not_x)
g = estimate_sources(src, W, glut=True)
p = estimate_sources(src, W, gap=True)
gb = estimate_sources(src, W, glut=True, gap=True)
demo = dict(S=S, credal=[lo, hi], credal_width=hi - lo, a=e.x.a, coefficients=coef, range=list(e.x.range()), width=e.x.width,
            width_by_type=by_type, width_by_type_rel=by_type_rel, width_by_source=by_src, width_by_source_rel=by_src_rel,
            total_symbols_kept=e.total.a, total_symbols_width=e.total.width, total_interval_arithmetic=list(ia),
            not_x=dict(a=e.not_x.a, range=list(e.not_x.range())),
            glut=dict(total=g.total.a, x_range=list(g.x.range()), x_width=g.x.width, notx_range=list(g.not_x.range()),
                      x_width_by_type=decompose(g.x, "type")),
            gap=dict(total=p.total.a, x_range=list(p.x.range()), x_width=p.x.width, notx_range=list(p.not_x.range())),
            glut_and_gap=dict(total=gb.total.a))
out["demo"] = demo
say("DEMO credal interval [%.3f, %.3f] width %.3f; S = %g" % (lo, hi, hi - lo, S))
say("DEMO p(x) =", e.x, "range [%.4f, %.4f] width %.4f" % (*e.x.range(), e.x.width))
say("DEMO width by type", {k: round(v, 4) for k, v in by_type.items()}, "relative", {k: round(v, 3) for k, v in by_type_rel.items()})
say("DEMO width by source", {k: round(v, 4) for k, v in by_src.items()}, "relative", {k: round(v, 3) for k, v in by_src_rel.items()})
say("DEMO p(x)+p(not x): symbols kept %.3f (width %.3g); interval arithmetic [%.3f, %.3f]" % (e.total.a, e.total.width, *ia))
say("DEMO glut reading: total %.3f; p(x) range [%.3f, %.3f] width %.3f; p(not x) range [%.3f, %.3f]" % (
    g.total.a, *g.x.range(), g.x.width, *g.not_x.range()))
say("DEMO gap reading: total %.3f; p(x) range [%.3f, %.3f] width %.3f" % (p.total.a, *p.x.range(), p.x.width))
say("DEMO p(not x) range [%.3f, %.3f]; glut and gap together: total %.4f" % (*e.not_x.range(), gb.total.a))

# ------------------------------------------------------------------ 2. random checks of Proposition 12.7.1 and of the reconciliation
rng = random.Random(20261003)
N = 20000
err = dict(range_vs_idm=0.0, coherence=0.0, width_vs_named=0.0, glut_width=0.0, glut_lower=0.0, gap_total=0.0,
           glut_total=0.0, single_symbol_idm=0.0, multi_source_range=0.0)
for _ in range(N):
    t, f, c, v, n = (rng.choice([0, 0, rng.randint(1, 12)]) for _ in range(5))
    extra = tuple(rng.choice([0, rng.randint(1, 6)]) for _ in range(rng.randint(0, 2)))
    Wr = rng.choice([0.5, 1.0, 2.0, 4.0])
    r = Reports(t, f, c, v, n, extra)
    Sr = t + f + c + v + n + sum(extra) + Wr
    est = estimate(r, Wr)
    l, u = credal_interval(r, Wr)
    err["range_vs_idm"] = max(err["range_vs_idm"], abs(est.x.lo - t / Sr), abs(est.x.hi - (Sr - f) / Sr), abs(est.x.lo - l), abs(est.x.hi - u))
    err["coherence"] = max(err["coherence"], abs(est.total.a - 1), est.total.width)
    named = (c + v + n + sum(extra) + Wr) / Sr
    err["width_vs_named"] = max(err["width_vs_named"], abs(est.x.width - named), abs((u - l) - named))
    gl = estimate(r, Wr, glut=True)
    err["glut_width"] = max(err["glut_width"], abs(gl.x.width - (named - c / Sr)))
    err["glut_lower"] = max(err["glut_lower"], abs(gl.x.lo - (t + c) / Sr), abs(gl.not_x.lo - (f + c) / Sr))
    err["glut_total"] = max(err["glut_total"], abs(gl.total.a - (1 + c / Sr)))
    gp = estimate(r, Wr, gap=True)
    err["gap_total"] = max(err["gap_total"], abs(gp.total.a - (1 - n / Sr)))
    if c == v == n == 0 and not any(extra):          # evidential triple / IDM of Corollary 2 with s = W
        err["single_symbol_idm"] = max(err["single_symbol_idm"], abs(est.x.lo - t / (t + f + Wr)), abs(est.x.width - Wr / (t + f + Wr)))
    # several sources: same range as the pooled reports
    k = rng.randint(2, 4)
    parts = [Reports(*(rng.randint(0, 5) for _ in range(5))) for _ in range(k)]
    tot = parts[0]
    for q in parts[1:]: tot = tot + q
    es = estimate_sources(parts, Wr)
    l2, u2 = credal_interval(tot, Wr)
    err["multi_source_range"] = max(err["multi_source_range"], abs(es.x.lo - l2), abs(es.x.hi - u2), abs(es.total.a - 1))
out["random_checks"] = dict(cases=N, seed=20261003, max_errors=err)
say("RANDOM %d cases, max errors" % N, {k: float("%.2g" % v) for k, v in err.items()})

# ------------------------------------------------------------------ 3. reconciliation: Theorem 1 on a normalised dual example, several I
# Theorem 1: I = U - L when (N) and (D) hold; with several named I the gap U - L is their total.
r = Reports(t=5, f=3, c=1, v=2, n=1)
est = estimate(r, 2.0)
l, u = credal_interval(r, 2.0)
recon = dict(reports=dict(t=5, f=3, c=1, v=2, n=1, W=2), S=14, L=l, U=u, gap=u - l, named=decompose(est.x, "type"),
             named_total=est.x.width, T=5 / 14, F=3 / 14, I_single=1 - 5 / 14 - 3 / 14)
gl = estimate(r, 2.0, glut=True)
recon["glut"] = dict(x_range=list(gl.x.range()), x_width=gl.x.width, notx_lower=gl.not_x.lo, total=gl.total.a)
out["reconciliation"] = recon
say("RECON L=%.4f U=%.4f gap=%.4f named total=%.4f; T=%.4f I=%.4f F=%.4f; by type %s" % (
    l, u, u - l, est.x.width, 5 / 14, 1 - 8 / 14, 3 / 14, {k: round(v, 4) for k, v in recon["named"].items()}))
say("RECON glut reading: p(x) range [%.4f, %.4f] width %.4f; lower of not x %.4f; total %.4f" % (*gl.x.range(), gl.x.width, gl.not_x.lo, gl.total.a))

# ------------------------------------------------------------------ 4. the acetaminophen/ASD case (one retrospective case)
summ = json.load(open(CASE + "/results/summary.json", encoding="utf-8"))
cum = list(csv.DictReader(open(CASE + "/results/cumulative.csv", encoding="utf-8")))
a4 = list(csv.DictReader(open(CASE + "/results/a4_classical_cumulative.csv", encoding="utf-8")))
coding = list(csv.DictReader(open(CASE + "/results/coding.csv", encoding="utf-8")))
fin = summ["final_main"]
# recompute the final state with rnel: association without identification = extra type I_1 (named I_A)
rA = Reports(t=fin["n_t"], f=fin["n_f"], c=fin["n_c"], v=fin["n_v"], extra=(fin["n_A"],))
eA = estimate(rA, 2.0)
wt = decompose(eA.x, "type")
recomputed = dict(lower=eA.x.lo, upper=eA.x.hi, width=eA.x.width, I_C=wt.get("C", 0), I_U=wt.get("U", 0), I_A=wt.get("I1", 0), I_G=wt.get("G", 0))
diff = max(abs(recomputed["lower"] - fin["lower"]), abs(recomputed["upper"] - fin["upper"]), abs(recomputed["width"] - fin["width"]),
           abs(recomputed["I_C"] - fin["w_I_C"]), abs(recomputed["I_U"] - fin["w_I_U"]), abs(recomputed["I_A"] - fin["w_I_A"]), abs(recomputed["I_G"] - fin["w_I_G"]))
ident = [x for x in coding if x["identification_design"] == "True" and x["type"] != "excluded"]
used = [x for x in coding if x["type"] != "excluded"]
case = dict(n_rows=summ["n_rows"], n_used=summ["n_used"], excluded=summ["excluded"], counts=summ["type_counts_final"], final=fin,
            recomputed_with_rnel=recomputed, max_diff_recomputed=diff,
            main_by_year=[{k: x[k] for k in ("year", "n_analyses", "lower", "upper", "width", "w_I_C", "w_I_U", "w_I_A", "w_I_G", "dominant")}
                          for x in cum if x["variant"] == "main"],
            S1_2026=[{k: x[k] for k in ("lower", "upper", "width", "w_I_U", "dominant")} for x in cum if x["variant"] == "S1_weak_as_undetermined" and x["year"] == "2026"][0],
            dominant_by_year_all_variants=summ["dominant_by_year_all_variants"],
            a4=a4, identification_rows=len(ident), identification_types={t_: sum(1 for x in ident if x["type"] == t_) for t_ in "tfcvA"},
            years=sorted({int(x["year"]) for x in used}))
out["case"] = case
say("CASE rows %d, used %d, counts %s" % (case["n_rows"], case["n_used"], case["counts"]))
say("CASE 2026 range [%.3f, %.3f] width %.3f; I_C %.3f I_U %.3f I_A %.3f I_G %.3f; recomputed with rnel, max diff %.1e" % (
    fin["lower"], fin["upper"], fin["width"], fin["w_I_C"], fin["w_I_U"], fin["w_I_A"], fin["w_I_G"], diff))
for x in case["main_by_year"]:
    say("CASE main", x["year"], "n", x["n_analyses"], "range [%s, %s] width %s; C %s U %s A %s G %s; dominant %s" % (
        x["lower"], x["upper"], x["width"], x["w_I_C"], x["w_I_U"], x["w_I_A"], x["w_I_G"], x["dominant"]))
say("CASE S1 2026", case["S1_2026"])
say("CASE identification rows", case["identification_rows"], case["identification_types"])
for x in a4: say("CASE A4", x)

# ------------------------------------------------------------------ 5. the simulation with known truth
sim = list(csv.DictReader(open(CASE + "/simulation/results/sim_summary.csv", encoding="utf-8")))
reg = open(CASE + "/simulation/results/posthoc_regret.txt", encoding="utf-8").read()
plan = open(CASE + "/simulation/00_PLAN_SIMULACION.md", encoding="utf-8").read()
simtab = {}
for x in sim:
    simtab.setdefault(x["pi_conf"], {})[x["policy"]] = dict(O1=float(x["O1_abs_error"]), O1_mcse=float(x["O1_mcse"]),
                                                          O2=float(x["O2_verdict_loss"]), share_tie=float(x["share_tie"]))
regret = {}
block = None
for ln in reg.splitlines():
    if ln.startswith("O1_abs_error"): block = "O1"
    elif ln.startswith("O2_verdict_loss"): block = "O2"
    m = re.match(r"\s+(.+?)\s+max ([0-9.]+)\s+mean ([0-9.]+)", ln)
    if m and block: regret.setdefault(block, {})[m.group(1)] = dict(max=float(m.group(2)), mean=float(m.group(3)))
best = {}
for pi, d in simtab.items():
    cand = {k: v["O1"] for k, v in d.items() if k not in ("oracle", "random", "typed (ties->MORE)")}
    best[pi] = min(cand, key=cand.get)
ties = [d["typed (ties->IDENT)"]["share_tie"] for d in simtab.values()]
mcse = [v["O1_mcse"] for d in simtab.values() for v in d.values()]
out["simulation"] = dict(table=simtab, best_O1_by_scenario=best, regret=regret, tie_share_range=[min(ties), max(ties)],
                         O1_mcse_range=[min(mcse), max(mcse)], worlds_per_scenario=int(re.search(r"(\d+) worlds per scenario", plan).group(1)),
                         seed=int(re.search(r"seed (\d+)", plan).group(1)),
                         plan_data_generating_mechanism=plan[plan.index("## Data-generating mechanism"):plan.index("## Estimand")])
for pi, d in simtab.items():
    say("SIM pi=%s" % pi, {k: d[k]["O1"] for k in ("typed (ties->IDENT)", "interval-learned", "always MORE", "always LARGER", "always IDENT")},
        "O2 typed %.4f learned %.4f" % (d["typed (ties->IDENT)"]["O2"], d["interval-learned"]["O2"]), "best:", best[pi])
say("SIM regret", regret)
say("SIM tie share range", [round(x, 3) for x in out["simulation"]["tie_share_range"]], "O1 MCSE range", out["simulation"]["O1_mcse_range"])

json.dump(out, open(os.path.join(RES, "stats_refined.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(os.path.join(RES, "stats_refined.out"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
assert all(v < 1e-12 for v in err.values()), err
assert diff < 5e-4, diff
print("ALL CHECKS PASSED")
