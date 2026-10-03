import numpy as np
from rnel.credal import to_neutrosophic, from_neutrosophic
from rnel import neutro_credal as nc

# Two events with lower and upper probabilities (L, U)
L = np.array([0.20, 0.55])
U = np.array([0.70, 0.90])
T, I, F = to_neutrosophic(L, U)                 # Theorem 1: T = L, I = U - L, F = 1 - U
for k in range(2):
    print(f"event {k}: (T, I, F) = ({T[k]:.2f}, {I[k]:.2f}, {F[k]:.2f})  sum = {T[k] + I[k] + F[k]:.2f}")
L2, U2 = from_neutrosophic(T, I, F)             # inverse map
print("round trip error:", float(np.abs(L2 - L).max() + np.abs(U2 - U).max()))

# Theorem 1(e): on a two-point frame every normalised dual assignment is coherent
NP2 = {"a": (0.3, 0.5, 0.2), "b": (0.2, 0.5, 0.3)}   # (D): T(b) = F(a), F(b) = T(a)
print("two-point frame:", nc.classify(NP2, outcomes=["a", "b"]))
