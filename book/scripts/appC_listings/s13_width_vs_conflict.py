from rnel import neutro_credal as nc
from rnel.credal import decide

lab = ["A", "B"]
cases = {                                            # members, evidence per member
    "weak, agreeing": ([(0.8, 0.2), (0.8, 0.2)], 0.5),
    "strong, agreeing": ([(0.8, 0.2), (0.8, 0.2)], 20.0),
    "strong, opposed": ([(0.9, 0.1), (0.1, 0.9)], 20.0),
}
for name, (members, scale) in cases.items():
    rep = nc.credal_diagnosis(members, labels=lab, evidence_scale=scale)
    s = rep["rnel"]["per_class"]["A"]
    print(f"{name:16} triple(A) = {tuple(round(v, 2) for v in rep['triples']['A'])}  "
          f"a = {s['a']:.2f}, b = {s['b']:.2f}, u = {s['u']:.2f}, c = {s['c']:.2f}  "
          f"-> {decide(s)[0]}")
