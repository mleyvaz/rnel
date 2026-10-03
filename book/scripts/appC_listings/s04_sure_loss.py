from rnel import neutro_credal as nc

# Theorem 3 on one event: sure loss iff T + F > 1, degree (T + F - 1)^+ / 2; I plays no role
for t in [(0.6, 0.2, 0.3), (0.7, 0.1, 0.6), (0.7, 0.9, 0.6), (0.9, 0.0, 0.9),
          (1.3, 0.0, 0.0), (-0.2, 0.5, 0.0)]:
    print(f"{str(t):17} avoids sure loss = {str(nc.avoids_sure_loss(t)):5}  "
          f"degree = {nc.sure_loss_degree(t):.3f}  coherent = {nc.is_coherent(t)}")

# Theorem 4 / Table 3: the interval [T, 1 - F] cannot see I
for t in [(0, 0, 0), (0, 1, 0), (0, 0.37, 0)]:
    print(f"{str(t):14} -> interval [{t[0]}, {1 - t[2]}]")

# Theorem 3(d): an underset T < 0 is corrected to 0 by the natural extension
print("natural extension of T = -0.2:",
      nc.natural_extension({"A": (-0.2, 0.5, 0.0)}, "A", outcomes=["A", "notA"], method="lp"))
