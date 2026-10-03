"""Numerical answers to the exercises of Part VI (Chapters 16-18). Output: solutions_part6.json (current directory, book edition).
The typed loss of Exercise 17.4 is computed with rnel.nn in a separate process (torch), as in numbers_torch.py."""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.getcwd()  # book edition: outputs go to the current working directory
sys.stdout.reconfigure(encoding="utf-8")

from rnel import Reports, rnel_tuple, fused_contradiction, sl_opinion  # noqa: E402
from rnel.sl import degree_of_conflict  # noqa: E402
from rnel.decide import Policy  # noqa: E402
from rnel import conflict as cf  # noqa: E402
from rnel import neutro_credal as nc  # noqa: E402
from rnel.off import retraction_nu, off_from_evidence  # noqa: E402

S, W = {}, 2.0


def put(k, v):
    S[k] = v if isinstance(v, (str, bool, list)) else float(v)


# 16.1
t = rnel_tuple(Reports(t=3, f=1, c=2, v=1, n=1), W)
for k, v in t.as_dict().items():
    put("16.1." + k, v)
put("16.1.total", t.total)
o = t.to_sl(); put("16.1.sl_b", o.b); put("16.1.sl_d", o.d); put("16.1.sl_u", o.u)
put("16.1.decision", Policy().decide(t)[0])
put("16.1.side_count", t.T * W / t.G); put("16.1.required", Policy().required_side_evidence())
# 16.2
x = fused_contradiction([Reports(t=8), Reports(f=8)], W); put("16.2.total", x.total); put("16.2.excess", x.total - 1)
# 16.3
tr = (0.9, 0.45, 0.45)
b, d, u, _ = retraction_nu(*tr); put("16.3.b", b); put("16.3.d", d); put("16.3.u", u)
put("16.3.paraconsistency", sum(tr) - 1); put("16.3.glut_lower", nc.to_glut_frame(tr).glut_lower())
# 16.4
for tr in ((0.5, 0.5, 0.5), (0.7, 0.2, 0.6), (0.2, 0.9, 0.1)):
    key = "16.4.%s" % ",".join(str(v) for v in tr)
    for fr in ("belnap", "disjoint", "minimal"):
        put("%s.%s" % (key, fr), nc.to_glut_frame(tr, fr).represents())
# 16.5
single, two = [Reports(t=4, f=4)], [Reports(t=4), Reports(f=4)]
put("16.5.single.C", fused_contradiction(single, W).C); put("16.5.two.C", fused_contradiction(two, W).C)
put("16.5.single.Cstar", cf.ConflictState.of_profile(single).c_star(W))
put("16.5.two.Cstar", cf.ConflictState.of_profile(two).c_star(W))
lo = 4 / 10; put("16.5.ci_lo", lo); put("16.5.ci_hi", 1 - 4 / 10)
# 16.6
prof = [(5, 1), (1, 5), (2, 2)]
st = cf.ConflictState.of_profile(prof)
put("16.6.R", st.R); put("16.6.S", st.S); put("16.6.Kw", st.Kw); put("16.6.Kb", st.Kb); put("16.6.Cstar", st.c_star(W))
put("16.6.check", st.Kb == min(st.R, st.S) - st.Kw)
ops = [sl_opinion(Reports(t=r, f=s_), W) for r, s_ in prof]
put("16.6.maxpair_DC", max(degree_of_conflict(ops[i], ops[j]) for i in range(3) for j in range(i + 1, 3)))
# 16.7  ledger: A 2 for, B 3 against, C 1 for; A retracted and counted once as anti-evidence
off = off_from_evidence(1 - 2, 3, W)
put("16.7.b", off.b); put("16.7.d", off.d); put("16.7.u", off.u); put("16.7.beta_ok", bool(off.beta_admissible))
put("16.7.r", -1.0); put("16.7.bound_r", -0.5 * W)
# 16.8
old, new = [], []
for tt in range(0, 11):
    tp = rnel_tuple(Reports(t=tt, f=3), W)
    if Policy(side_gate=False).decide(tp)[0] == "F":
        old.append(tt)
    if Policy().decide(tp)[0] == "F":
        new.append(tt)
put("16.8.old_refuted_t", old); put("16.8.new_refuted_t", new)
# 17.1
e = (5.0, 0, 0, 0, 0)
Sg = sum(e) + W; put("17.1.T", e[0] / Sg); put("17.1.G", W / Sg)
# 17.2
e = (2.0, 2.0, 4.0, 0.0, 0.0); Sg = sum(e) + W
put("17.2.T", e[0] / Sg); put("17.2.F", e[1] / Sg); put("17.2.C", e[2] / Sg); put("17.2.G", W / Sg)
put("17.2.sl_u", 1 - (e[0] + e[1]) / Sg); put("17.2.dissonance_binary", 2 * min(e[0], e[1]) / (e[0] + e[1] + W))
# 17.3
prof = [(4, 1), (1, 4), (2, 2)]
st = cf.ConflictState.of_profile(prof)
R, Sx = st.R, st.S
put("17.3.dissonance", 2 * min(R, Sx) / (R + Sx + W)); put("17.3.Cstar", st.c_star(W)); put("17.3.Cw", st.c_within(W))
put("17.3.single_Cstar", cf.ConflictState.of_profile([(7, 7)]).c_star(W))
# 17.4 (closed form; checked against rnel.nn in a torch subprocess below)
e = [1.0, 1.0, 6.0, 0.0, 0.0]; a = [v + W / 5 for v in e]
put("17.4.loss_C", math.log(sum(a)) - math.log(a[2])); put("17.4.loss_T", math.log(sum(a)) - math.log(a[0]))
code = ("import torch,sys;from rnel.nn import typed_evidential_loss as L;"
        "e=torch.tensor([[1.,1.,6.,0.,0.]]);print(L(e,torch.tensor([2])).item(),L(e,torch.tensor([0])).item())")
out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.expanduser("~"))
lc, lt = map(float, out.stdout.split())
put("17.4.torch_check", abs(lc - S["17.4.loss_C"]) < 1e-5 and abs(lt - S["17.4.loss_T"]) < 1e-5)
# 17.5
reads = [(0.9, 0.05, 0.05), (0.1, 0.85, 0.05), (0.2, 0.1, 0.7)]
reps = [Reports(t=5 * en, f=5 * co) for en, co, _ in reads]
x = fused_contradiction(reps, W)
put("17.5.T", x.T); put("17.5.F", x.F); put("17.5.G", x.G); put("17.5.C", x.C)
# 18.1
for name, (a1, a2) in (("split", (8, 3)), ("close", (6, 5))):
    o1, o2 = sl_opinion(Reports(t=a1, f=10 - a1), W), sl_opinion(Reports(t=a2, f=10 - a2), W)
    put("18.1.%s.b1" % name, o1.b); put("18.1.%s.d1" % name, o1.d); put("18.1.%s.b2" % name, o2.b); put("18.1.%s.d2" % name, o2.d)
    put("18.1.%s.C" % name, degree_of_conflict(o1, o2))

json.dump(S, open(os.path.join(OUT, "solutions_part6.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for k in S:
    v = S[k]
    print("%-36s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))
