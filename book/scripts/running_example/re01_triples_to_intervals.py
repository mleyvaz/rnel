"""Step 1. From the adviser's triples to credal intervals (Definition 2, Theorem 1, Theorem 4)."""
from re_common import COURSES, NEUTRO, INTUITIONISTIC, intervals, nc

print("Step 1: neutrosophic triples of the four courses and their betting intervals [T, 1 - F]")
print(f"{'course':6} {'(T, I, F)':18} {'T+I+F':>6} {'interval':>14} {'width':>6} {'stated I':>8}")
for c in COURSES:
    T, I, F = NEUTRO[c]
    l, u = T, 1 - F
    print(f"{c:6} {str(NEUTRO[c]):18} {T + I + F:6.1f} [{l:.1f}, {u:.1f}]{'':4} {u - l:6.1f} {I:8.1f}")

print("\nTheorem 1 (inverse map T = L, I = U - L, F = 1 - U): the normalised triple that the classical")
print("frame actually carries for each course, i.e. the natural extension on {pass, fail}:")
for c in COURSES:
    t = nc.natural_extension({"pass": NEUTRO[c]}, "pass", outcomes=["pass", "fail"], method="lp")
    tn = tuple(round(v, 3) for v in t)
    print(f"  {c}: stated {NEUTRO[c]} -> carried {tn}")

print("\nIntuitionistic and neutrosophic forms give the same intervals (Theorem 4(b): kernel = I direction):")
iv_n, iv_i = intervals("neutrosophic"), intervals("intuitionistic")
for c in COURSES:
    print(f"  {c}: neutrosophic {iv_n[c]}  intuitionistic {iv_i[c]}  equal = {iv_n[c] == iv_i[c]}")
assert all(abs(iv_n[c][0] - iv_i[c][0]) < 1e-12 and abs(iv_n[c][1] - iv_i[c][1]) < 1e-12 for c in COURSES)
