"""Step 2. Coherence and sure loss (Theorems 1 and 3), course by course and jointly."""
import itertools

from re_common import COURSES, NEUTRO, nc

print("Step 2a: Theorem 3 on each course (sure loss iff T + F > 1, degree (T + F - 1)^+ / 2)")
for c in COURSES:
    t = NEUTRO[c]
    print(f"  {c} {t}: T + F = {t[0] + t[2]:.1f}  degree = {nc.sure_loss_degree(t):.3f}  "
          f"coherent = {nc.is_coherent({'pass': t}, outcomes=['pass', 'fail'], method='lp')}")

# Joint frame: 16 pass/fail patterns; event "pass course k" and event "graduates" (all four passed)
outcomes = ["".join(p) for p in itertools.product("01", repeat=4)]
passk = {c: frozenset(o for o in outcomes if o[k] == "1") for k, c in enumerate(COURSES)}
grad = frozenset(["1111"])
marg = {passk[c]: NEUTRO[c] for c in COURSES}

print("\nStep 2b: the four course triples as one assessment on the 16-atom frame")
print("  avoids sure loss:", nc.avoids_sure_loss(marg, outcomes=outcomes),
      " coherent:", nc.is_coherent(marg, outcomes=outcomes, method="lp"))
ne = nc.natural_extension(marg, grad, outcomes=outcomes, method="lp")
print("  natural extension to 'graduates' (T, I, F) =", tuple(round(v, 3) for v in ne))
assert abs(ne[0] - 0.0) < 1e-9 and abs(ne[2] - 0.5) < 1e-9

print("\nStep 2c: the adviser adds a direct triple for 'graduates'")
for gt in [(0.6, 0.1, 0.3), (0.0, 0.6, 0.4), (0.0, 0.5, 0.5)]:
    a = dict(marg)
    a[grad] = gt
    d = nc.sure_loss_degree(a, outcomes=outcomes)
    asl = nc.avoids_sure_loss(a, outcomes=outcomes)
    coh = nc.is_coherent(a, outcomes=outcomes, method="lp") if asl else False
    line = f"  graduates {gt}: degree of sure loss = {d:.3f}  avoids = {asl}  coherent = {coh}"
    if asl and not coh:
        corr = nc.natural_extension(a, grad, outcomes=outcomes, method="lp")
        line += f"  -> natural extension {tuple(round(v, 3) for v in corr)}"
    print(line)
