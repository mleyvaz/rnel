"""Credal tools for neutrosophic triples: a runnable walk-through.

    python examples/neutro_credal_demo.py

Needs numpy and scipy (pip install "rnel[credal]").
"""
from rnel import neutro_credal as nc


def fmt(t):
    return "(" + ", ".join(f"{v:.3f}" for v in t) + ")"


# Three classifiers, three classes ------------------------------------------------------------------
members = [(0.7, 0.2, 0.1), (0.6, 0.3, 0.1), (0.1, 0.8, 0.1)]
labels = ["A", "B", "C"]
rep = nc.credal_diagnosis(members, labels=labels, events={"A or C": ["A", "C"]})

print("Per-class triples (T, I, F) of the credal set spanned by the classifiers:")
for k, t in rep["triples"].items():
    print(f"  {k}: {fmt(t)}   interval [{t[0]:.2f}, {1 - t[2]:.2f}]")
print("Walley status of the triples:", rep["status"])
T, I, F = rep["events"]["A or C"]
print(f"Natural extension of 'A or C': {fmt((T, I, F))}  ->  P(A or C) in [{T:.2f}, {1 - F:.2f}]")
print("Decision by maximality:", rep["decision"])
print("Why is there uncertainty? (a support, b rejection, u absence, c conflict between sources)")
for k, s in rep["rnel"]["per_class"].items():
    print(f"  {k}: " + ", ".join(f"{x}={s[x]:.2f}" for x in "abuc"))
print(f"  multiclass C* = {rep['rnel']['C_star']:.3f}")

# Quality checks on hand-made triples ---------------------------------------------------------------
print("\nSure loss of a single triple (Theorem 3):")
for t in [(0.6, 0.2, 0.3), (0.7, 0.1, 0.6)]:
    print(f"  {fmt(t)}: avoids sure loss = {nc.avoids_sure_loss(t)}, degree = {nc.sure_loss_degree(t):.3f}")

tri = [(0.3, 0.5, 0.2), (0.3, 0.2, 0.5)]
chk = nc.singleton_intervals_check(tri)
print(f"\nSingletons {tri}: Delta = {chk['delta']:.2f}, coherent = {chk['coherent']} "
      f"(I of outcome {chk['I_above_delta']} exceeds the truth deficit; Theorem 2)")
print("Coherent correction (I never increases):", [fmt(t) for t in nc.coherent_correction(tri)])

print("\nImprecise Dirichlet model, counts (6, 3, 1), s = 2:")
for k, t in zip(labels, nc.idm_triple([6, 3, 1], s=2)):
    print(f"  {k}: {fmt(t)}")

# The paraconsistent zone on the glut frame (Theorem 8) ---------------------------------------------
t = (0.8, 0.3, 0.7)
g = nc.to_glut_frame(t)
print(f"\nTriple {t}: sure loss degree on {{A, not A}} = {nc.sure_loss_degree(t):.2f}; "
      f"on the glut frame represented = {g.represents()}, least glut mass = {g.glut_lower():.2f}")
print("N-norm (independent) as a natural extension on the glut frame:",
      fmt(nc.glut_conjunction((0.6, 0.2, 0.3), (0.5, 0.4, 0.2), "independent")))
