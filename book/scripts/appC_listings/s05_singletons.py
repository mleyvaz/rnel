import numpy as np
from rnel import neutro_credal as nc


def show(name, tri):
    chk = nc.singleton_intervals_check(tri)
    print(f"{name}: Delta = {chk['delta']:.2f}, sum I = {chk['I'].sum():.2f}, "
          f"avoids sure loss = {chk['avoids_sure_loss']}, coherent = {chk['coherent']}, "
          f"I_i > Delta for {chk['I_above_delta']}, Delta > rest for {chk['delta_above_rest']}")


# Three outcomes, one triple each (Theorem 2)
good = [(0.30, 0.20, 0.50), (0.25, 0.20, 0.55), (0.20, 0.25, 0.55)]
bad = [(0.50, 0.40, 0.10), (0.20, 0.10, 0.70), (0.10, 0.10, 0.80)]
show("good", good)
show("bad ", bad)

# Natural extension of the composite event {0, 2}: closed form (Theorem 2(c)) and LP
for m in ("closed", "lp"):
    print(f"NP^E({{0, 2}}) [{m:6}] =", tuple(round(v, 4) for v in nc.natural_extension(good, [0, 2], method=m)))

# Correction of the incoherent system: I can only fall (Theorem 1(c))
fix = nc.coherent_correction(bad)
print("corrected:", np.round(fix, 3).tolist())
show("fixed", fix)
