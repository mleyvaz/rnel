"""Step 3. Lifting with gluts: each course (Theorem 8), all twelve components (Corollary 3(a), minimal
lifting with 13 atoms) and the product lifting (Corollary 3(c), 256 atoms)."""
import numpy as np

from re_common import (COURSES, NEUTRO, INTUITIONISTIC, lp_min, minimal_lifting_patterns,
                       product_lifting, product_constraints, nc)

print("Step 3a: each course on the 4-atom minimal lifting (Theorem 8)")
for c in COURSES:
    g = nc.to_glut_frame(NEUTRO[c])
    print(f"  {c} {NEUTRO[c]}: faithful = {g.represents()}  envelope = "
          f"{tuple(round(v, 6) for v in g.envelope())}  least glut mass = {g.glut_lower():.3f}")

print("\nThe intuitionistic pair of SA, (0.6, 0.4), is the triple (0.6, 0, 0.4); the neutrosophic one is (0.6, 0.2, 0.4).")
gi, gn = nc.to_glut_frame((0.6, 0.0, 0.4)), nc.to_glut_frame(NEUTRO["SA"])
print(f"  lower P(E_I): intuitionistic {gi.lower(gi.event('I')):.3f}  neutrosophic {gn.lower(gn.event('I')):.3f}"
      "  -> different credal sets on the lifting, same interval [0.6, 0.6] on {pass, fail}")

# ---- minimal lifting of the 12 components (Corollary 3(a))
x = np.array([v for c in COURSES for v in NEUTRO[c]])          # order: DE(T,I,F), SA(T,I,F), ...
pats = minimal_lifting_patterns(12)                              # 13 atoms x 12 events
A, b = -pats.T, -x
env = [lp_min(pats[:, j], A, b, len(pats)) for j in range(12)]
print(f"\nStep 3b: minimal lifting, {len(pats)} atoms; largest envelope error = {np.max(np.abs(np.array(env) - x)):.1e}")
I_idx = [1, 4, 7, 10]
mean_I = pats[:, I_idx].mean(axis=1)
lo_min = lp_min(mean_I, A, b, len(pats))
print(f"  lower P(random-selection event over the four I events) = {lo_min:.3f}  (not 0.15)")
print("  reason: every atom misses at most one of the 12 events, so it lies in at least 3 of the 4 I events")
assert abs(lo_min - 0.75) < 1e-9

# ---- product lifting (Corollary 3(c))
n, E = product_lifting()
triples = NEUTRO
A, b = product_constraints(E, triples)
errs = [abs(lp_min(E[c][X], A, b, n) - triples[c][j]) for c in COURSES for j, X in enumerate("TIF")]
print(f"\nStep 3c: product lifting, {n} atoms; largest envelope error over the 12 components = {max(errs):.1e}")
mean_I = sum(E[c]["I"] for c in COURSES) / 4
lo_prod = lp_min(mean_I, A, b, n)
print(f"  lower P(random-selection event over the four I events) = {lo_prod:.3f}  (= mean of the I's, Proposition 2(c))")
assert abs(lo_prod - 0.15) < 1e-9
glut_any = np.zeros(n)
for c in COURSES:
    glut_any = np.maximum(glut_any, E[c]["T"] * E[c]["F"])
print(f"  least mass on 'some course is a glut' = {lp_min(glut_any, A, b, n):.3f}")
