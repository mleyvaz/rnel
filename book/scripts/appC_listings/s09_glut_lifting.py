from rnel import neutro_credal as nc

# Theorem 8(a): the minimal lifting represents every triple of the cube
for t in [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.8, 0.3, 0.7), (0.70, 0.10, 0.60)]:
    g = nc.to_glut_frame(t)
    print(f"{str(t):17} two-atom sure loss = {nc.sure_loss_degree(t):.2f} | lifting: "
          f"represents = {g.represents()}, envelope = {tuple(round(v, 3) for v in g.envelope())}, "
          f"least glut mass = {g.glut_lower():.2f}")

g = nc.to_glut_frame((0.8, 0.3, 0.7))
print("atoms:", g.atoms)
print("patterns (E_T, E_I, E_F):", g.patterns.astype(int).tolist())
print("extreme points (rows attain T, I, F):", g.extreme_points().round(2).tolist())

# Theorem 8(d): the faithful regions of the Belnap and disjoint frames
for t in [(0.5, 0.3, 0.2), (0.6, 0.3, 0.6), (0.6, 0.5, 0.6)]:
    print(f"{str(t):16} belnap: {str(nc.GlutLifting(t, 'belnap').represents()):5}  "
          f"disjoint: {nc.GlutLifting(t, 'disjoint').represents()}")
