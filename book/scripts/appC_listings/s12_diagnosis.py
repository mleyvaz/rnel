from rnel import neutro_credal as nc

members = [(0.7, 0.2, 0.1), (0.6, 0.3, 0.1), (0.1, 0.8, 0.1)]   # three classifiers, classes A, B, C
rep = nc.credal_diagnosis(members, labels=["A", "B", "C"], events={"A or C": ["A", "C"]})

for k, t in rep["triples"].items():
    print(f"{k}: (T, I, F) = ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})   interval [{t[0]:.2f}, {1 - t[2]:.2f}]")
print("status:", rep["status"])
print("A or C:", tuple(round(v, 3) for v in rep["events"]["A or C"]))
print("decision (" + rep["rule"] + "):", rep["decision"])
for k, s in rep["rnel"]["per_class"].items():
    print(f"{k}: " + ", ".join(f"{x} = {s[x]:.2f}" for x in "abuc"))
print("multiclass C* =", round(rep["rnel"]["C_star"], 3))
for rule in ("interval_dominance", "e_admissible"):
    print(rule, nc.credal_diagnosis(members, labels=["A", "B", "C"], rule=rule)["decision"])
