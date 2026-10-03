from rnel import neutro_credal as nc

x, y = (0.6, 0.2, 0.3), (0.5, 0.4, 0.2)


def r(t):
    return tuple(round(v, 4) for v in t)


# Theorem 9: N-norms as natural extensions on the 16-atom product of two minimal liftings
for dep in ("independent", "none", "comonotone"):
    print(f"{dep:12}", r(nc.glut_conjunction(x, y, dep)))
print("none (LP)   ", r(nc.glut_conjunction(x, y, "none", method="lp")))

# Theorem 7(a), normalised inputs: on the classical frame the credal conjunction under
# independence is [T1 T2, u1 u2] with u = 1 - F; the algebraic N-norm has the same T and F
x, y = (0.6, 0.2, 0.2), (0.5, 0.3, 0.2)
(T1, I1, F1), (T2, I2, F2) = x, y
lo, hi = T1 * T2, (1 - F1) * (1 - F2)
alg = nc.glut_conjunction(x, y, "independent")
print("classical credal (T, I, F) =", r((lo, hi - lo, 1 - hi)), "| algebraic N-norm =", r(alg))
print("classical credal I =", round(hi - lo, 4), "| algebraic I =", round(alg[1], 4),
      "| surplus =", round(alg[1] - (hi - lo), 4), "| I1 F2 + I2 F1 =", round(I1 * F2 + I2 * F1, 4))
