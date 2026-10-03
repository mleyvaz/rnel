"""Corollary 3 (v1.1): the imprecise Dirichlet model on the Belnap frame {t, f, b, n}.
E_T = {t, b}, E_F = {f, b}, E_I = {n}. Typed counts n_t, n_f, n_b, n_n, prior strength s.
Checks, on random counts, by linear programming over the IDM credal set
    M = {P on {t,f,b,n} : P(x) >= n_x/(N+s) for every atom x},
(a) lower envelopes of E_T, E_F, E_I equal the closed forms, and the triple lies in the faithful region of Theorem 6(d);
(b) T + I + F = 1 - Delta + n_b/(N+s), with Delta = s/(N+s);
(c) min_{P in M} [P(b) - P(n)] = (n_b - n_n - s)/(N+s), so T + F > 1 iff n_b > n_n + s iff P(b) > P(n) on all of M;
(d) (T, I, F, Delta) and s recover the counts.
Also records the worked example and the Dirichlet tail probabilities quoted in Section 7.4.
"""
import json
import os

import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta

RESULTS = os.getcwd()  # book edition: outputs go to the current working directory
rng = np.random.default_rng(20261001)
ATOMS = ["t", "f", "b", "n"]
EV = {"T": [0, 2], "F": [1, 2], "I": [3]}


def lp_min(c, lo):
    """min c.P over the IDM credal set: P >= lo, sum P = 1."""
    r = linprog(c, A_eq=[np.ones(4)], b_eq=[1.0], bounds=[(l, 1) for l in lo], method="highs")
    assert r.status == 0
    return r.fun


def closed(n, s):
    N = sum(n)
    D = N + s
    T, F, I = (n[0] + n[2]) / D, (n[1] + n[2]) / D, n[3] / D
    return T, I, F, s / D


err = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
glut_cases = zone_mismatch = 0
CASES = 3000
for _ in range(CASES):
    n = rng.integers(0, 15, size=4).astype(float)
    s = float(rng.choice([0.5, 1.0, 2.0, 3.0]))
    N = n.sum()
    lo = n / (N + s)
    T, I, F, Delta = closed(n, s)
    for X, val in (("T", T), ("F", F), ("I", I)):
        c = np.zeros(4)
        c[EV[X]] = 1.0
        err["a"] = max(err["a"], abs(lp_min(c, lo) - val))
    assert T + I <= 1 + 1e-12 and F + I <= 1 + 1e-12          # faithful region of Theorem 6(d)
    err["b"] = max(err["b"], abs((T + I + F) - (1 - Delta + n[2] / (N + s))))
    m = lp_min(np.array([0, 0, 1.0, -1.0]), lo)
    err["c"] = max(err["c"], abs(m - (n[2] - n[3] - s) / (N + s)))
    zone = T + F > 1 + 1e-12
    glut_cases += zone
    zone_mismatch += zone != (n[2] > n[3] + s)
    zone_mismatch += zone != (m > 1e-12)
    Dn = s / Delta
    nb = (T + I + F - 1 + Delta) * Dn
    rec = np.array([T * Dn - nb, F * Dn - nb, nb, I * Dn])
    err["d"] = max(err["d"], float(np.max(np.abs(rec - n))))

assert max(err.values()) < 1e-9 and zone_mismatch == 0, (err, zone_mismatch)

# worked example (counts from an evidence audit, single coder, used as illustration) and Dirichlet tails
examples = {"running_pooled": [4, 3, 1, 3], "endurance_af_pooled": [15, 8, 3, 8], "football_pooled": [18, 0, 2, 4]}
ex_out = {}
for k, n in examples.items():
    T, I, F, Delta = closed(n, 2.0)
    ex_out[k] = {"counts_t_f_b_n": n, "s": 2, "T": round(T, 3), "I": round(I, 3), "F": round(F, 3),
                 "Delta": round(Delta, 3), "T_plus_F": round(T + F, 3),
                 "robust_glut": bool(n[2] > n[3] + 2),
                 # precise Dirichlet posterior with uniform prior (alpha = n + 1): P(p_b > p_n)
                 "P_glut_uniform_prior": round(float(beta.sf(0.5, n[2] + 1, n[3] + 1)), 3),
                 # range over the IDM prior family (s = 2): Beta(n_b, n_n + s) to Beta(n_b + s, n_n)
                 "P_glut_IDM_lower": round(float(beta.sf(0.5, max(n[2], 1e-9), n[3] + 2)), 3),
                 "P_glut_IDM_upper": round(float(beta.sf(0.5, n[2] + 2, max(n[3], 1e-9))), 3)}

out = {"cases": CASES, "max_errors": err, "paraconsistent_cases": int(glut_cases),
       "zone_mismatches": int(zone_mismatch), "examples": ex_out}
os.makedirs(RESULTS, exist_ok=True)
json.dump(out, open(os.path.join(RESULTS, "v11_belnap_idm.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
print("v11_belnap_idm: OK")
