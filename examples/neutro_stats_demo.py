"""Credal output and neutrosophic output of the same reports, side by side."""
from rnel.neutro_stats import credal_interval, decompose, estimate, estimate_sources, interval_sum
from rnel.tuple import Reports

sources = {"trial A": Reports(t=4, c=2), "cohort B": Reports(f=3, c=1, n=2)}
pooled = Reports(t=4, f=3, c=3, n=2)

print("credal (IDM, set-valued reports):  p(x) in [%.3f, %.3f]" % credal_interval(pooled))
e = estimate_sources(sources)
print("neutrosophic, several I:           p(x) =", e.x)
print("  width by type:  ", {k: round(v, 3) for k, v in decompose(e.x, "type", relative=True).items()})
print("  width by source:", {k: round(v, 3) for k, v in decompose(e.x, "source", relative=True).items()})
print("  p(x)+p(not x): symbols kept -> %.3f ; interval arithmetic -> [%.3f, %.3f]"
      % ((e.total.a,) + interval_sum(e.x, e.not_x)))
g = estimate_sources(sources, glut=True)
print("glut reading:  p(x)+p(not x) = %.3f  (no probability on {x, not x} gives this)" % g.total.a)
