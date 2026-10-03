from itertools import combinations
from rnel import neutro_credal as nc


def dual_np(L, n):
    """Normalised dual NP from a lower probability on proper events: T = L(A), F = L(not A)."""
    full = frozenset(range(n))
    return {A: (L[A], 1 - L[A] - L[full - A], L[full - A]) for A in L}


events = [frozenset(c) for k in (1, 2) for c in combinations(range(3), k)]

# Counterexample C1: T = 0.4 on singletons, 0.3 on pairs
C1 = dual_np({A: 0.4 if len(A) == 1 else 0.3 for A in events}, 3)
print("C1:", nc.classify(C1, outcomes=3), "| degree of sure loss =", round(nc.sure_loss_degree(C1, outcomes=3), 6))

# Counterexample C2 (Table 1): T({0}) = 0.5 but T({0,1}) = 0.2
L = {frozenset({0}): 0.5, frozenset({1}): 0.0, frozenset({2}): 0.0,
     frozenset({0, 1}): 0.2, frozenset({0, 2}): 0.5, frozenset({1, 2}): 0.0}
C2 = dual_np(L, 3)
print("C2:", nc.classify(C2, outcomes=3), "| degree of sure loss =", round(nc.sure_loss_degree(C2, outcomes=3), 6))

# Natural extension and coherent correction (Theorem 1(c))
fix = nc.coherent_correction(C2, outcomes=3)
print("event    assessed (T, I, F)    corrected (T, I, F)")
for A in sorted(C2, key=lambda a: (len(a), sorted(a))):
    a, b = C2[A], fix[A]
    print(f"{str(sorted(A)):8} ({a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f})    ({b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f})")
print("corrected assessment coherent:", nc.is_coherent(fix, outcomes=3))
