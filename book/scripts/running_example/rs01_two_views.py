"""Worked example of the RNEL chapter: two views that disagree versus two views that agree.

Profile 1: views (8, 1, 1) and (1, 8, 1) over three classes.  Profile 2: (4.5, 4.5, 1) twice.
W = 2, proposition 'class = 0'. Uses rnel.credal.evidential_state and rnel.credal.c_star_multiclass
(branch feat/neutro-credal-tools), plus the glut lifting of rnel.neutro_credal and a direct check of
the credal-gap identity D = (beta - alpha) K_b.
"""
import os
import sys

import numpy as np

RNEL_SRC = os.environ.get("RNEL_SRC")  # optional; default: rnel from PYTHONPATH
if RNEL_SRC:
    sys.path.insert(0, RNEL_SRC)
from rnel import credal, neutro_credal as nc  # noqa: E402

W = 2.0
profiles = {"disagree": np.array([[8, 1, 1], [1, 8, 1]], float),
            "agree": np.array([[4.5, 4.5, 1], [4.5, 4.5, 1]], float)}

for name, E in profiles.items():
    Ev = E[:, None, :]                       # shape (views, instances=1, classes)
    st = credal.evidential_state(Ev, 0, W=W)
    cm = float(credal.c_star_multiclass(Ev, W=W)[0])
    v = {k: float(np.atleast_1d(st[k])[0]) for k in ("a", "b", "u", "c", "R", "S", "Kw", "Kb", "T", "I", "F")}
    R, S = v["R"], v["S"]
    print(f"[{name}] views {E.tolist()}")
    print(f"  pooled (R, S) = ({R:g}, {S:g}), K_w = {v['Kw']:g}, K_b = {v['Kb']:g}")
    print(f"  SL cumulative fusion (b, d, u) = ({R/(R+S+W):.3f}, {S/(R+S+W):.3f}, {W/(R+S+W):.3f})")
    print(f"  state (a, b, u, c) = ({v['a']:.3f}, {v['b']:.3f}, {v['u']:.3f}, {v['c']:.3f})  sum = "
          f"{v['a']+v['b']+v['u']+v['c']:.3f}")
    print(f"  multiclass sup-norm C* = {cm:.3f}")
    # credal set over the sources: per-view probability of class 0 (normalised view evidence)
    p_views = E[:, 0] / E.sum(axis=1)
    p_fused = E.sum(axis=0)[0] / E.sum()
    print(f"  class-0 probability per view = {p_views.round(3).tolist()}, width over views = "
          f"{p_views.max() - p_views.min():.3f}; fused (pooled) probability = {p_fused:.3f}")
    # paraconsistent reading T = a + c, I = u + c, F = b + c and the classical frame / glut lifting
    T, I, F = v["T"], v["I"], v["F"]
    print(f"  paraconsistent reading (T, I, F) = ({T:.3f}, {I:.3f}, {F:.3f}); T + F = {T + F:.3f}")
    print(f"    classical frame: degree of sure loss = {nc.sure_loss_degree((T, I, F)):.3f}")
    g = nc.to_glut_frame((T, I, F))
    print(f"    minimal glut lifting: faithful = {g.represents()}, least glut mass = {g.glut_lower():.3f}"
          f" (= c - u = {v['c'] - v['u']:.3f})")
    # credal-gap identity (Theorem 2 of the conflict manuscript) on the one-vs-rest counts
    r, s = E[:, 0], E[:, 1:].sum(axis=1)
    for alpha, beta in [(0.0, 1.0), (0.2, 0.7)]:
        phi = lambda rr, ss: max(p * rr + (1 - p) * ss for p in (alpha, beta))
        gap = sum(phi(ri, si) for ri, si in zip(r, s)) - phi(r.sum(), s.sum())
        print(f"    credal gap on [{alpha}, {beta}] = {gap:.3f}  vs (beta - alpha) K_b = {(beta - alpha) * v['Kb']:.3f}")
    print()

# The binary pair of Proposition 8.3 in the RNEL paper: A = (10, 0), B = (0, 10)
Eb = np.array([[10, 0], [0, 10]], float)[:, None, :]
stb = credal.evidential_state(Eb, 0, W=W)
print(f"[binary pair (10,0), (0,10)] C* = {float(np.atleast_1d(stb['c'])[0]):.3f}; "
      f"single source (10,10): C* = {float(np.atleast_1d(credal.evidential_state(np.array([[10, 10]], float)[:, None, :], 0, W=W)['c'])[0]):.3f}")
