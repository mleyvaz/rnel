"""T3 (sure loss = T+F-1; off-values), T4 (affine readings: uniqueness + blindness),
T5b (retraction to SL through the saturating evidential chart; two readings of T+F>1).
"""
import os
import sys

import numpy as np
import sympy as sp

from common import credal_nonempty, sure_loss_degree, natural_extension, is_coherent, save



def tf_to_subjective(T, F):
    """Exact map from the saturating measures (T, F) to SL (b, d, u) (Proposition 1').
    Book edition: copied verbatim from nel/opinion.py of the NEL repository (no external import)."""
    denom = 1.0 - T * F
    if denom <= 0:
        raise ValueError("T = F = 1 has no subjective-logic image")
    return T * (1 - F) / denom, F * (1 - T) / denom, (1 - T) * (1 - F) / denom


rng = np.random.default_rng(11)
out = {}
a, na = frozenset({0}), frozenset({1})


def single(T, I, F):
    """Single event A on the binary frame, with complement duality."""
    return {a: (T, I, F), na: (F, I, T)}


# T3. sure-loss degree of a single event -------------------------------------------------------
err = 0.0
for _ in range(4000):
    T, I, F = rng.uniform(0, 1, 3)
    d = sure_loss_degree(single(T, I, F), 2)
    err = max(err, abs(d - max(0.0, (T + F - 1) / 2)))
    # betting argument: buying A at T and not-A at F pays 1 for sure, costs T+F
    assert (T + F > 1 + 1e-9) == (not credal_nonempty(single(T, I, F), 2))
out["single_event_delta_formula_max_err"] = err
out["single_event_I_invisible"] = True  # I never enters the LP constraints
assert err < 1e-8

# off-values (Smarandache over/under-sets)
over = single(1.3, 0.2, 0.0)
under = single(-0.2, 0.3, 0.5)
out["overset_T=1.3"] = {"nonempty": credal_nonempty(over, 2), "delta": sure_loss_degree(over, 2)}
ne_under = natural_extension(under, 2, a)
out["underset_T=-0.2"] = {"nonempty": credal_nonempty(under, 2), "coherent": is_coherent(under, 2),
                          "natural_extension_T": ne_under}
assert not credal_nonempty(over, 2) and abs(sure_loss_degree(over, 2) - 0.3) < 1e-8
assert credal_nonempty(under, 2) and not is_coherent(under, 2) and abs(ne_under) < 1e-9

# T4. affine readings ---------------------------------------------------------------------------
T, I, F, al, be = sp.symbols("T I F alpha beta", real=True)
s = T + I + F - 1
l = T + al * s
u = T + I + be * s
M = sp.Matrix([[sp.diff(l, v) for v in (T, I, F)], [sp.diff(u, v) for v in (T, I, F)]])
ker = M.nullspace()
out["affine_kernel_generic"] = [str(list(k)) for k in ker]
out["kernel_walley(alpha=0,beta=-1)"] = str(list(M.subs({al: 0, be: -1}).nullspace()[0]))
out["kernel_T_T+I(alpha=0,beta=0)"] = str(list(M.subs({al: 0, be: 0}).nullspace()[0]))
# vertex constraints forcing alpha = 0, beta = -1 (endpoints must stay in [0,1])
cube = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
feasible = []
for A_ in np.linspace(-1, 1, 201):
    for B_ in np.linspace(-2, 1, 301):
        ok = True
        for (x, y, z) in cube:
            ss = x + y + z - 1
            lv, uv = x + A_ * ss, x + y + B_ * ss
            if not (-1e-12 <= lv <= 1 + 1e-12 and -1e-12 <= uv <= 1 + 1e-12):
                ok = False
                break
        if ok:
            feasible.append((round(float(A_), 6), round(float(B_), 6)))
out["affine_parameters_with_endpoints_in_[0,1]"] = feasible
assert feasible == [(0.0, -1.0)]
# with the Walley reading, l <= u  <=>  T + F <= 1 (checked on random points)
pts = rng.uniform(0, 1, (20000, 3))
assert np.all((pts[:, 0] <= 1 - pts[:, 2]) == (pts[:, 0] + pts[:, 2] <= 1))

# collapses (neutrality / ignorance)
walley = lambda t, i, f: (t, 1 - f)          # noqa: E731
tti = lambda t, i, f: (t, t + i)             # noqa: E731
out["collapse_walley"] = {str(p): walley(*p) for p in [(0, 0, 0), (0, 1, 0), (0, .37, 0)]}
out["collapse_T_T+I"] = {str(p): tti(*p) for p in [(0, 0, 0), (0, 0, 1)]}

# T5b. retraction to SL via the saturating chart T = r/(r+W), F = s/(s+W) ------------------------
res = {"cases": 0, "identity_err": 0.0, "zone_mismatch": 0, "paraconsistent_cases": 0}
for _ in range(20000):
    W = float(rng.uniform(0.5, 5))
    r, q = rng.exponential(5, 2)
    Ts, Fs = r / (r + W), q / (q + W)
    b, d, uu = tf_to_subjective(Ts, Fs)
    res["identity_err"] = max(res["identity_err"], abs(b - r / (r + q + W)),
                              abs(d - q / (r + q + W)), abs(uu - W / (r + q + W)))
    res["zone_mismatch"] += (Ts + Fs > 1) != (r * q > W * W)
    if Ts + Fs > 1:
        res["paraconsistent_cases"] += 1
        # betting reading: sure loss; evidential reading: non-empty IDM interval [b, b+u]
        assert sure_loss_degree(single(Ts, 0.0, Fs), 2) > 0
        assert 0 <= b <= b + uu <= 1
    # base-rate orbit of the projected probability equals the credal interval
    grid = b + np.linspace(0, 1, 11) * uu
    assert abs(grid.min() - b) < 1e-12 and abs(grid.max() - (1 - d)) < 1e-12
    res["cases"] += 1
assert res["identity_err"] < 1e-9 and res["zone_mismatch"] == 0
out["retraction_saturating_chart"] = res

save("t3_t4_t5_readings.json", out)
print("T3/T4/T5b OK")
for k in ("single_event_delta_formula_max_err", "overset_T=1.3", "underset_T=-0.2",
          "affine_kernel_generic", "kernel_walley(alpha=0,beta=-1)",
          "affine_parameters_with_endpoints_in_[0,1]", "retraction_saturating_chart"):
    print(" ", k, out[k])
