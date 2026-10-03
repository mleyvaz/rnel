from rnel import neutro_credal as nc

# Corollary 4: n_? unclassifiable answers act as extra prior strength, s' = s + n_?
regions = {"North": (112, 38, 20), "Centre": (95, 30, 45), "South": (60, 52, 8)}
s = 2.0
print("region   (T, I, F) of 'satisfied'   [T, 1-F]          indeterminate dropped")
for name, (sat, notsat, unk) in regions.items():
    T, I, F = nc.idm_triple([sat, notsat], s=s + unk)[0]
    T0, I0, F0 = nc.idm_triple([sat, notsat], s=s)[0]
    print(f"{name:7}  ({T:.3f}, {I:.3f}, {F:.3f})   [{T:.3f}, {1 - F:.3f}]    [{T0:.3f}, {1 - F0:.3f}]")

# As s -> 0 the interval tends to Manski's worst-case bounds [112/170, 132/170]
print("Manski:", round(112 / 170, 4), round(132 / 170, 4))
for s in (2.0, 1.0, 0.1):
    T, I, F = nc.idm_triple([112, 38], s=s + 20)[0]
    print(f"s = {s}: [{T:.4f}, {1 - F:.4f}]")
