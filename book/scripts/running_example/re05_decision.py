"""Step 5. Decisions: (a) will Jenifer graduate this semester? (b) where should one hour of tutoring go?

(a) uses interval dominance and E-admissibility of rnel.neutro_credal on the two outcomes
{graduates, does not graduate}, with the graduation interval of each dependence model.
(b) is a what-if: raise the chance of passing one course by 0.1 (T + 0.1, F - 0.1, I unchanged) and
recompute the graduation interval under the dependence models of Table 8.
"""
import numpy as np

from re_common import COURSES, NEUTRO, nc

def models(tr):
    T = {c: tr[c][0] for c in COURSES}
    U = {c: 1 - tr[c][2] for c in COURSES}
    return {
        "no assumption": (max(0.0, sum(T.values()) - 3), min(U.values())),
        "independence": (np.prod(list(T.values())), np.prod(list(U.values()))),
        "comon. in subject": (min(T["DE"], T["SA"]) * min(T["FM"], T["SM"]),
                              min(U["DE"], U["SA"]) * min(U["FM"], U["SM"])),
        "comonotone": (min(T.values()), min(U.values())),
    }

print("Step 5a: decision between 'graduates' and 'does not graduate'")
for name, (l, u) in models(NEUTRO).items():
    tri = [(l, max(0.0, u - l), 1 - u), (1 - u, max(0.0, u - l), l)]   # triples of the two outcomes
    dom = nc.interval_dominance(tri, labels=["graduates", "not"])
    ead = nc.e_admissible(tri, labels=["graduates", "not"])
    print(f"  {name:18} P(graduates) in [{l:.3f}, {u:.3f}]  not interval-dominated: {dom}  E-admissible: {ead}")

print("\nStep 5b: what-if, one course raised by 0.1 (lower, upper probability of graduating)")
base = models(NEUTRO)
print(f"  {'raised':8}" + "".join(f"{m:>22}" for m in base))
print(f"  {'none':8}" + "".join(f"   [{l:.3f}, {u:.3f}]".rjust(22) for l, u in base.values()))
gain = {}
for c in COURSES:
    tr = dict(NEUTRO)
    T, I, F = tr[c]
    tr[c] = (round(T + 0.1, 10), I, round(max(0.0, F - 0.1), 10))
    res = models(tr)
    gain[c] = res
    print(f"  {c:8}" + "".join(f"   [{l:.3f}, {u:.3f}]".rjust(22) for l, u in res.values()))
for m in base:
    lo_best = max(COURSES, key=lambda c: (gain[c][m][0], gain[c][m][1]))
    hi_best = max(COURSES, key=lambda c: (gain[c][m][1], gain[c][m][0]))
    print(f"  {m:18}: best lower bound from {lo_best}, best upper bound from {hi_best}")
