import numpy as np
from rnel import neutro_credal as nc

counts, s = np.array([6, 3, 1]), 2.0
tri = nc.idm_triple(counts, s=s)                 # Corollary 2: T_i = n_i/(N+s), I_i = s/(N+s)
print("IDM triples:", np.round(tri, 4).tolist())
print("coherent:", nc.is_coherent(tri), "| Delta =", round(nc.singleton_intervals_check(tri)["delta"], 4))

N = counts.sum()
for A in ([0], [0, 1], [1, 2]):
    T, I, F = nc.natural_extension(tri, A, method="lp")
    nA = counts[A].sum()
    print(f"A = {A}: LP [{T:.4f}, {1 - F:.4f}]   Walley IDM [{nA / (N + s):.4f}, {(nA + s) / (N + s):.4f}]")
