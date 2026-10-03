import numpy as np
from rnel import neutro_credal as nc

# A credal classifier from IDM counts (Section 12.3)
for counts in ([9, 7, 2], [18, 14, 4]):
    tri = nc.idm_triple(counts, s=2)
    print(counts, "triples", np.round(tri, 3).tolist(),
          "-> interval dominance:", nc.interval_dominance(tri, labels=["1", "2", "3"]))

# Four classes, credal set = convex hull of two member distributions
members = [(0.45, 0.00, 0.25, 0.30),
           (0.20, 0.50, 0.25, 0.05)]
lab = ["A", "B", "C", "D"]
print("interval dominance:", nc.interval_dominance(members=members, labels=lab))
print("maximality:        ", nc.maximality(members=members, labels=lab))
print("E-admissibility:   ", nc.e_admissible(members=members, labels=lab))
