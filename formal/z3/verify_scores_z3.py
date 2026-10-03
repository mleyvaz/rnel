"""Verificación automática (Z3, aritmética real exacta) de las afirmaciones
del paper "funciones de score neutrosóficas = tasas base ocultas".

Cada afirmación se comprueba negándola: si Z3 responde `unsat`, no existe
contraejemplo y la afirmación es VÁLIDA para todos los reales que cumplen
las hipótesis. Si responde `sat`, se imprime el contraejemplo.

Ejecutar:  python verify_scores_z3.py
"""
from z3 import Reals, And, Not, Solver, sat, unsat, Q

T, I, F, a, lam = Reals("T I F a lam")
T1, I1, F1, T2, I2, F2 = Reals("T1 I1 F1 T2 I2 F2")


def unit(*xs):
    return And(*[And(x >= 0, x <= 1) for x in xs])


def S(t, f, a_):                 # S_a = (1-a) T + a (1-F)
    return (1 - a_) * t + a_ * (1 - f)


def S_lam(t, i, f, a_, l_):      # S_{a,lambda} = S_a - lambda * I
    return S(t, f, a_) - l_ * i


def classic(t, i, f):            # score clásica (2 + T - I - F) / 3
    return (2 + t - i - f) / 3


def check(name, hyp, claim, expect_valid=True):
    s = Solver()
    s.add(hyp, Not(claim))
    r = s.check()
    valid = (r == unsat)
    tag = "VÁLIDA" if valid else "FALSA"
    ok = "OK" if valid == expect_valid else "¡INESPERADO!"
    print(f"[{ok}] {name}: {tag}")
    if r == sat:
        m = s.model()
        print("      contraejemplo:", {str(d): str(m[d]) for d in m.decls()})
    return valid == expect_valid


results = []

# C1. Descomposición (identidad para todos los reales)
results.append(check(
    "C1 (2+T-I-F)/3 = (2/3)*S_{1/2} + (1-I)/3",
    True, classic(T, I, F) == Q(2, 3) * S(T, F, Q(1, 2)) + (1 - I) / 3))

# C1b. La score clásica es transformación afín positiva de S_{1/2, lambda=1/2}
results.append(check(
    "C1b (2+T-I-F)/3 = (2/3)*S_{1/2,1/2} + 1/3",
    True, classic(T, I, F) == Q(2, 3) * S_lam(T, I, F, Q(1, 2), Q(1, 2)) + Q(1, 3)))

# C2. Caso normalizado: se reduce a (1+2T)/3
norm = And(unit(T, I, F), T + I + F == 1)
results.append(check(
    "C2 normalizado => (2+T-I-F)/3 = (1+2T)/3",
    norm, classic(T, I, F) == (1 + 2 * T) / 3))

# C3. Caso normalizado: la score clásica ordena exactamente como T
normAB = And(unit(T1, I1, F1, T2, I2, F2), T1 + I1 + F1 == 1, T2 + I2 + F2 == 1)
results.append(check(
    "C3 normalizado => [classic(A) > classic(B) <=> T_A > T_B]",
    normAB, (classic(T1, I1, F1) > classic(T2, I2, F2)) == (T1 > T2)))

# C4. T-F ordena igual que S_{1/2} (sin hipótesis: todos los reales)
results.append(check(
    "C4 [T_A-F_A > T_B-F_B] <=> [S_1/2(A) > S_1/2(B)]",
    True, (T1 - F1 > T2 - F2) == (S(T1, F1, Q(1, 2)) > S(T2, F2, Q(1, 2)))))

# C5. Fórmula única: S_a queda entre T y 1-F (hueco y glut)
lo_hi = And(unit(T, F, a))
results.append(check(
    "C5 hueco (T+F<=1): T <= S_a <= 1-F",
    And(lo_hi, T + F <= 1), And(T <= S(T, F, a), S(T, F, a) <= 1 - F)))
results.append(check(
    "C5 glut (T+F>=1): 1-F <= S_a <= T  y  S_a = T - a(T+F-1)",
    And(lo_hi, T + F >= 1),
    And(1 - F <= S(T, F, a), S(T, F, a) <= T, S(T, F, a) == T - a * (T + F - 1))))

# C6. Caso normalizado: S_{a,lambda} = T + (a - lambda) I  -> colapsa a T si a = lambda
results.append(check(
    "C6 normalizado => S_{a,lam} = T + (a-lam) I",
    And(norm, unit(a, lam)), S_lam(T, I, F, a, lam) == T + (a - lam) * I))

# C7. Tasa base imprecisa: A domina a B para TODO a en [aL,aU]
#     <=> domina en los dos extremos (linealidad en a)
aL, aU = Reals("aL aU")
dom_ext = And(S(T1, F1, aL) >= S(T2, F2, aL), S(T1, F1, aU) >= S(T2, F2, aU))
results.append(check(
    "C7 dominancia en extremos => dominancia para a intermedio",
    And(unit(T1, F1, T2, F2, aL, aU, a), aL <= a, a <= aU, dom_ext),
    S(T1, F1, a) >= S(T2, F2, a)))

# --- Afirmaciones que DEBEN ser falsas (control: Z3 debe hallar contraejemplo)
anyAB = unit(T1, I1, F1, T2, I2, F2)
results.append(check(
    "F1 (control) sin normalizar, classic ordena como T",
    anyAB, (classic(T1, I1, F1) > classic(T2, I2, F2)) == (T1 > T2),
    expect_valid=False))
results.append(check(
    "F2 (control) sin normalizar, classic ordena como S_1/2",
    anyAB, (classic(T1, I1, F1) > classic(T2, I2, F2)) == (S(T1, F1, Q(1, 2)) > S(T2, F2, Q(1, 2))),
    expect_valid=False))

print()
print("TODAS COMO SE ESPERABA" if all(results) else "HAY RESULTADOS INESPERADOS")
