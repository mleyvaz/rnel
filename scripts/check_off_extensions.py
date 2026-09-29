"""Numerical check of Section 10 (Over/Under/Off extensions) of Smarandache & Leyva-Vázquez,
"The Neutrosophic (T, I, N, F), Multi-Uncertainty, n-Valued Refined, Refined Evidential Logics with Neutrosophic
Probability, and their Over/Under/Off Extensions - as generalizations of the Subjective Logic" (submitted version).
Preprint: https://doi.org/10.20944/preprints202609.2584

Checks Theorems 10.3-10.6, recomputes the four applications and the statements on the retraction nu as they appear
in the text, and reports PASS / MISMATCH for each. Run:  python scripts/check_off_extensions.py
Exit code 1 if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rnel import sl  # noqa: E402
from rnel.off import (complement_discount, off_from_evidence, off_rnel_tuple, projected_via_nu,  # noqa: E402
                      retraction_nu, scaled_and, signed_comultiply, signed_multiply, typed_discount)
from rnel.operators import mass, tinf_and, tinf_not, tinf_or  # noqa: E402
from rnel.tuple import Reports, fused_contradiction  # noqa: E402

rng = np.random.default_rng(20260928)
LOG = []


def check(name, ok, detail=""):
    LOG.append(("PASS" if ok else "MISMATCH", name, detail))


def note(name, detail):
    LOG.append(("NOTE", name, detail))


def close(x, y, tol=1e-3):
    return abs(x - y) <= tol


# ------------------------------------------------------------------ Theorem 10.3 (Beta semantics, fusion)
W, a = 2.0, 0.5
err_fusion, err_mean, n_ok = 0.0, 0.0, 0
for _ in range(20000):
    rA, sA, rB, sB = rng.uniform(-0.99, 20, 4)
    A, B = off_from_evidence(rA, sA, W, a), off_from_evidence(rB, sB, W, a)
    F = off_from_evidence(rA + rB, sA + sB, W, a)
    k = A.u + B.u - A.u * B.u
    if abs(k) < 1e-9:
        continue
    b = (A.b * B.u + B.b * A.u) / k
    u = A.u * B.u / k
    err_fusion = max(err_fusion, abs(b - F.b), abs(u - F.u))
    if F.beta_admissible:
        n_ok += 1
        err_mean = max(err_mean, abs(F.projected - (F.r + W * a) / (F.r + F.s + W)))
check("Th 10.3(3): SL cumulative fusion = addition of signed evidence (20,000 pairs)", err_fusion < 1e-9, f"max error {err_fusion:.1e}")
check("Th 10.3(1): P equals the Beta mean in the Beta region", err_mean < 1e-12, f"max error {err_mean:.1e} over {n_ok} cases")
x = off_from_evidence(-1.5, 0, W, a)
check("Th 10.3(1): no Beta density when r <= -Wa", (not x.beta_admissible) and x.projected <= 0, f"r=-1.5: P={x.projected:.3f}")

# ------------------------------------------------------------------ Theorem 10.4 (priority products on off-tuples)
keys = ["T", "I", "N", "F"]
e_dm = e_mass = e_proj = e_neg = e_scale = 0.0
for _ in range(3000):
    X = dict(zip(keys, rng.uniform(-1, 2, 4)))
    Y = dict(zip(keys, rng.uniform(-1, 2, 4)))
    lhs, rhs = tinf_not(tinf_and(X, Y)), tinf_or(tinf_not(X), tinf_not(Y))
    e_dm = max(e_dm, max(abs(lhs[k] - rhs[k]) for k in keys))
    e_mass = max(e_mass, abs(mass(tinf_and(X, Y)) - mass(X) * mass(Y)), abs(mass(tinf_or(X, Y)) - mass(X) * mass(Y)))
    e_neg = max(e_neg, max(abs(tinf_not(tinf_not(X))[k] - X[k]) for k in keys))
for _ in range(2000):  # SL plane (N = 0, S = 1), off-opinions allowed
    bx, ux, by, uy = rng.uniform(-0.5, 1.5, 4)
    X = {"T": bx, "I": ux, "N": 0.0, "F": 1 - bx - ux}
    Y = {"T": by, "I": uy, "N": 0.0, "F": 1 - by - uy}
    c = tinf_and(X, Y)
    m = signed_multiply((bx, 1 - bx - ux, ux), (by, 1 - by - uy, uy), a=0.0)
    d = tinf_or(X, Y)
    cm = signed_comultiply((bx, 1 - bx - ux, ux), (by, 1 - by - uy, uy), a=1.0)
    e_proj = max(e_proj, abs(c["T"] - m[0]), abs(c["F"] - m[1]), abs(d["T"] - cm[0]), abs(d["F"] - cm[1]))
for _ in range(2000):  # over-plane S = sigma
    sigma = rng.uniform(1.1, 3)
    X = dict(zip(keys, rng.dirichlet(np.ones(4)) * sigma))
    Y = dict(zip(keys, rng.dirichlet(np.ones(4)) * sigma))
    e_scale = max(e_scale, abs(mass(scaled_and(X, Y, sigma)) - sigma))
check("Th 10.4: De Morgan on off-tuples in [-1,2]^4", e_dm < 1e-11, f"{e_dm:.1e}")
check("Th 10.4: total mass multiplicative", e_mass < 1e-11, f"{e_mass:.1e}")
check("Th 10.4: negation is an involution", e_neg == 0, f"{e_neg:.1e}")
check("Th 10.4: projection = SL multiplication (a=0) / comultiplication (a=1) on the off-plane", e_proj < 1e-11, f"{e_proj:.1e}")
check("Th 10.4: scaled operators keep S = sigma", e_scale < 1e-11, f"{e_scale:.1e}")

# ------------------------------------------------------------------ Theorem 10.5 and the compromised relay
x = {"T": 0.8, "F": 0.1, "U_evidence": 0.1}
y = typed_discount(-0.2, x, "U_platform")
rep = {"T": -0.160, "F": -0.020, "U_evidence": -0.020, "U_platform": 1.200}
check("Relay: off-tuple with p = -0.2", all(close(y[k], rep[k]) for k in rep), str({k: round(v, 3) for k, v in y.items()}))
b, d = y["T"], y["F"]
u = 1 - b - d
P = b + 0.5 * u
check("Relay: projection (b,d,u) = (-0.160,-0.020,1.180), P = 0.430", close(b, -0.16) and close(u, 1.18) and close(P, 0.43), f"P={P:.3f}")
honest = typed_discount(0.7, x, "U_platform")
Ph = honest["T"] + 0.5 * (1 - honest["T"] - honest["F"])
check("Relay: honest platform p = 0.7 gives P = 0.745", close(Ph, 0.745), f"P={Ph:.3f}")
cb, cd, cu = complement_discount(-0.2, 0.8, 0.1, 0.1)
check("Relay, Remark: evidence on the complement with trust |p| = 0.2 gives (0.02, 0.16, 0.82), P = 0.430",
      close(cb, 0.02) and close(cd, 0.16) and close(cu, 0.82) and close(cb + 0.5 * cu, 0.43),
      f"(b,d,u)=({cb:.3f},{cd:.3f},{cu:.3f}), P={cb + 0.5 * cu:.3f}")
e_gap = 0.0
for _ in range(2000):  # P_off - P_complement = q (b + d)(2a - 1)
    q_, a_ = rng.uniform(0, 1), rng.uniform(0, 1)
    bb, dd = rng.dirichlet(np.ones(3))[:2]
    uu = 1 - bb - dd
    p_off = -q_ * bb + a_ * (1 + q_ * bb + q_ * dd)
    p_cmp = q_ * dd + a_ * (1 - q_ * dd - q_ * bb)
    e_gap = max(e_gap, abs((p_off - p_cmp) - q_ * (bb + dd) * (2 * a_ - 1)))
check("Relay, Remark: P_off - P_complement = q(b + d)(2a - 1)", e_gap < 1e-12, f"{e_gap:.1e}")

# ------------------------------------------------------------------ Theorem 10.6 (closed form, refined tuples)
def closed_and(x, y):
    Lx, Ly = np.cumsum(x), np.cumsum(y)
    out = Lx * Ly
    out[1:] -= Lx[:-1] * Ly[:-1]
    return out


e_tel = 0.0
for _ in range(3000):
    xx, yy = rng.uniform(-1, 2, 6), rng.uniform(-1, 2, 6)
    e_tel = max(e_tel, abs(closed_and(xx, yy).sum() - xx.sum() * yy.sum()))
check("Th 10.6: closed-form conjunction keeps total mass multiplicative for real components", e_tel < 1e-11, f"{e_tel:.1e}")

# ------------------------------------------------------------------ Application: fake reviews (Section 10.3)
r_rep, fake, s, neutral, unread, W = 40, 25, 10, 30, 5, 2
sl_drop = sl.Opinion.from_evidence(r_rep - fake, s, W)
check("Fake reviews: SL dropping them gives (0.556,0.370,0.074), P=0.593",
      close(sl_drop.b, 0.556) and close(sl_drop.d, 0.370) and close(sl_drop.projected, 0.593),
      f"({sl_drop.b:.3f},{sl_drop.d:.3f},{sl_drop.u:.3f}), P={sl_drop.projected:.3f}")
t_paper = r_rep - fake - fake
T, I, N, F = t_paper / 87, (unread + W) / 87, neutral / 87, s / 87
score = (2 + T + N - I - F) / 4
check("Fake reviews: tuple (-0.115,0.080,0.345,0.115), score 0.509 (normalised by the original 87 reports)",
      close(T, -0.115) and close(score, 0.509), f"score={score:.3f}")
r_remaining = r_rep - fake            # Definition 10.2: r_rep counts the reports that remain after retraction
check("Fake reviews: Definition 10.2, r = r_rep - r_ret = (40 - 25) - 25 = -10", r_remaining - fake == t_paper,
      f"r = {r_remaining - fake}")
tot = t_paper + s + neutral + unread + W
T2, I2, N2, F2 = t_paper / tot, (unread + W) / tot, neutral / tot, s / tot
score2 = (2 + T2 + N2 - I2 - F2) / 4
check("Fake reviews: normalising by the signed total (37) gives s = 0.520", tot == 37 and close(score2, 0.520),
      f"total={tot}, score={score2:.3f}")
check("Fake reviews: seller A (0.509) still ranks above seller B (0.480)", score > 0.480, f"{score:.3f} > 0.480")

# ------------------------------------------------------------------ Application: RAG with retracted sources (10.6)
table = {0: (0.357, 0.357, 0.686, 0.119, 0.119, 0.048, 0.500, 0.500),
         1: (0.156, 0.469, 0.503, 0.156, 0.156, 0.063, 0.344, 0.407),
         2: (-0.227, 0.682, 0.0, 0.227, 0.227, 0.091, 0.045, 0.273)}
for k, row in table.items():
    t = 15 - 10 * k
    x = off_rnel_tuple(t=t, f=15, v=5, n=5, W=2)
    A = sl.Opinion.from_evidence(max(t, 0), 0)
    B = sl.Opinion.from_evidence(0, 15)
    C = sl.degree_of_conflict(A, B) if t > 0 else 0.0
    P_paper_formula = x["T"] + 0.5 * (x["U"] + x["N"] + x["G"])
    P_sl = sl.Opinion.from_evidence(15 - 5 * k, 15).projected
    got = (x["T"], x["F"], C, x["U"], x["N"], x["G"], P_paper_formula, P_sl)
    check(f"RAG, {k} retracted: table row", all(close(g, r, 2e-3) for g, r in zip(got, row)),
          ", ".join(f"{g:.3f}" for g in got))
    x_withC = dict(x, C=C)
    P_nu = projected_via_nu(x_withC)
    expected_nu = {0: 0.500, 1: 0.396, 2: 0.045}[k]
    check(f"RAG, {k} retracted: P through nu (C included, Definition 4.4) = {expected_nu:.3f}",
          close(P_nu, expected_nu, 2e-3), f"{P_nu:.3f}")

# ------------------------------------------------------------------ Application: over-vote (10.7)
R, rA, rB, ind, W = 1000, 1030, 20, 10, 2
NP = (rA / R, ind / R, rB / R)
refined = (rA / R, ind / R, W / R, rB / R)
o = sl.Opinion.from_evidence(rA, rB, W)
check("Over-vote: NP(A) = (1.03, 0.01, 0.02)", all(close(g, r) for g, r in zip(NP, (1.03, 0.01, 0.02))), str(NP))
check("Over-vote: refined (1.030, 0.010, 0.002, 0.020)", all(close(g, r) for g, r in zip(refined, (1.03, 0.01, 0.002, 0.02))), str(refined))
check("Over-vote: SL (0.979, 0.019, 0.002), P = 0.980", close(o.b, 0.979) and close(o.d, 0.019) and close(o.projected, 0.980),
      f"({o.b:.3f},{o.d:.3f},{o.u:.4f}), P={o.projected:.3f}")

# ------------------------------------------------------------------ Claims about the retraction nu
Rst, r, s, W = 100, 10, 5, 2
b, d, u, _ = retraction_nu(r / Rst, W / Rst, s / Rst)
o = sl.Opinion.from_evidence(r, s, W)
check("Section 10.3: the condition sigma > 1 is needed (sigma < 1, R=100: nu differs from SL)",
      not (close(b, o.b) and close(u, o.u)), f"nu=({b:.3f},{d:.3f},{u:.3f}) vs SL ({o.b:.3f},{o.d:.3f},{o.u:.3f})")
Rst = 10
b, d, u, _ = retraction_nu(r / Rst, W / Rst, s / Rst)
check("Section 10.3: when sigma > 1 (R=10), nu returns SL's opinion", close(b, o.b) and close(u, o.u),
      f"nu=({b:.3f},{d:.3f},{u:.3f})")
x = fused_contradiction([Reports(t=10), Reports(f=10)])
fused = sl.cumulative_fusion(sl.Opinion.from_evidence(10, 0), sl.Opinion.from_evidence(0, 10))
b, d, u, _ = retraction_nu(x.T, x.U + x.N + x.G, x.F)
check("Section 8.2: nu applied to (T, F, G), after removing C, returns SL's fused opinion",
      close(b, fused.b) and close(d, fused.d) and close(u, fused.u),
      f"nu=({b:.3f},{d:.3f},{u:.3f}) vs SL fusion ({fused.b:.3f},{fused.d:.3f},{fused.u:.3f})")
check("Section 8.2: fused contradiction C = 0.579 for 10 reports for and 10 against", close(x.C, 0.579), f"C={x.C:.3f}")

# ------------------------------------------------------------------ report
w = max(len(n) for _, n, _ in LOG)
for status, name, detail in LOG:
    print(f"{status:9s} {name:<{w}}  {detail}")
n_bad = sum(s == 'MISMATCH' for s, _, _ in LOG)
print(f"\n{sum(s == 'PASS' for s, _, _ in LOG)} PASS, {n_bad} MISMATCH")
sys.exit(1 if n_bad else 0)
