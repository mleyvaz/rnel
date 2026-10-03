"""v0.3 remark: the symmetric reading [(1+D-I)/2, (1+D+I)/2], D = T - F.

Checks that it is the affine reading of Theorem 4(a) with alpha = beta = -1/2,
that its kernel is the direction (1, 0, 1) (equal increases of T and F),
that it leaves [0,1] on the cube, and the worked values used in the remark.
"""
import json
from pathlib import Path

import numpy as np
import sympy as sp


def sym_reading(T, I, F):
    D = T - F
    return (1 + D - I) / 2, (1 + D + I) / 2


def affine(T, I, F, alpha, beta):
    s = T + I + F - 1
    return T + alpha * s, T + I + beta * s


out = {}
rng = np.random.default_rng(20260930)

# 1. identity with alpha = beta = -1/2 on the whole cube
X = rng.uniform(0, 1, size=(100_000, 3))
err = 0.0
for T, I, F in X:
    a = np.array(sym_reading(T, I, F))
    b = np.array(affine(T, I, F, -0.5, -0.5))
    err = max(err, float(np.abs(a - b).max()))
assert err < 1e-12
out["max_error_vs_alpha_beta_minus_half"] = err

# 2. kernel of the linear part
t, i, f = sp.symbols("t i f")
l, u = sym_reading(t, i, f)
M = sp.Matrix([[sp.diff(l, v) for v in (t, i, f)], [sp.diff(u, v) for v in (t, i, f)]])
ker = M.nullspace()
assert len(ker) == 1
k = list(ker[0])
k = [x / k[0] for x in k]
assert k == [1, 0, 1]
out["kernel_direction"] = [int(x) for x in k]

# 3. agreement with [T, T+I] = [T, 1-F] on the normalised plane
errp = 0.0
for _ in range(20_000):
    T, I = rng.dirichlet(np.ones(3))[:2]
    F = 1 - T - I
    lo, up = sym_reading(T, I, F)
    errp = max(errp, abs(lo - T), abs(up - (T + I)))
assert errp < 1e-12
out["max_error_on_normalised_plane"] = errp

# 4. worked values
cases = {
    "(0.7,0.2,0.3)": (0.7, 0.2, 0.3),
    "(0.7,0.2,0.5)": (0.7, 0.2, 0.5),
    "(0,1,1)": (0.0, 1.0, 1.0),
    "(0,0,0)": (0.0, 0.0, 0.0),
    "(0.2,0.3,0.2)": (0.2, 0.3, 0.2),
    "(0.5,0.3,0.5)": (0.5, 0.3, 0.5),
}
out["worked"] = {k: [round(x, 4) for x in sym_reading(*v)] for k, v in cases.items()}
assert out["worked"]["(0.7,0.2,0.3)"] == [0.6, 0.8]
assert out["worked"]["(0.7,0.2,0.5)"] == [0.5, 0.7]
assert out["worked"]["(0,1,1)"] == [-0.5, 0.5]
assert out["worked"]["(0,0,0)"] == [0.5, 0.5]
# equal support and rejection added: same interval
assert out["worked"]["(0.2,0.3,0.2)"] == out["worked"]["(0.5,0.3,0.5)"]

# 5. share of the cube where the reading leaves [0,1]
lo = (1 + X[:, 0] - X[:, 2] - X[:, 1]) / 2
up = (1 + X[:, 0] - X[:, 2] + X[:, 1]) / 2
out["share_outside_unit_interval"] = round(float(np.mean((lo < 0) | (up > 1))), 4)

Path.cwd().joinpath("v03_symmetric_reading.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
