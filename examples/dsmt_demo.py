"""rnel.dsmt demo: Zadeh's example under every rule, and the dispute that the rules treat differently.

Run:  python examples/dsmt_demo.py
"""
import sys

from rnel import Reports, fused_contradiction
from rnel import conflict as cf
from rnel.dsmt import Frame, RULES, binary_frame, bba_to_tuple, conflict, dsmc, reports_to_bba, tbm

sys.stdout.reconfigure(encoding="utf-8")

# 1. Zadeh's example (Shafer's model): where does the conflict K = 0.99 go?
fr = Frame(["A", "B", "C"], model="shafer")
m1, m2 = fr.bba({"A": 0.9, "C": 0.1}), fr.bba({"B": 0.9, "C": 0.1})
print("Zadeh, K =", round(conflict(fr, [m1, m2]), 4))
for name, rule in RULES.items():
    print(f"  {name:13s}", fr.pretty(rule(fr, [m1, m2]), 4))
free = Frame(["A", "B", "C"])
print("  DSmC (free)  ", free.pretty(dsmc(free, [free.bba({"A": 0.9, "C": 0.1}), free.bba({"B": 0.9, "C": 0.1})]), 4))

# 2. Two sources in dispute vs one balanced source (book, Proposition 16.5), W = 2
sh = binary_frame("shafer")
A, B, coin = (reports_to_bba(r, frame=sh) for r in (Reports(t=8), Reports(f=8), Reports(t=8, f=8)))
print("\nDispute (8,0)+(0,8) vs one balanced source (8,8)")
print("  one balanced source:", sh.pretty(coin, 3))
for name, rule in RULES.items():
    print(f"  dispute, {name:13s}", sh.pretty(rule(sh, [A, B]), 3))
print("  dispute, TBM          ", sh.pretty(tbm(sh, [A, B]), 3))
fb = binary_frame("free")
glut = dsmc(fb, [reports_to_bba(Reports(t=8)), reports_to_bba(Reports(f=8))])
print("  dispute, DSmC (free)  ", fb.pretty(glut, 3), "-> as an RNEL tuple:", {k: round(v, 3) for k, v in bba_to_tuple(glut).as_dict().items()})
print("  RNEL: C (Def. 8.4) =", round(fused_contradiction([Reports(t=8), Reports(f=8)]).C, 3),
      " C* =", round(cf.c_star([(8, 0), (0, 8)]), 3))
