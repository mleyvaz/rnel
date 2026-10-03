"""T2 (singleton NP = probability intervals, neutrosophic form of properness /
reachability / natural extension) and T5a (IDM = SL = evidential NP).

Singleton NP with T_i + I_i + F_i = 1 gives intervals l_i = T_i, u_i = 1 - F_i = T_i + I_i.
Let tau = sum T_i and delta0 = 1 - tau (truth deficit).
 (a) proper  <=>  0 <= delta0 <= sum_i I_i
 (b) reachable (coherent) <=> for all i:  I_i <= delta0 <= sum_{j != i} I_j
 (c) if reachable: L(A) = T_A + (delta0 - I_{not A})^+,  U(A) = T_A + min(I_A, delta0)
All checked against LP on random instances.
"""
import numpy as np

from common import events, complement, credal_nonempty, is_coherent, natural_extension, save

rng = np.random.default_rng(7)
out = {"proper_mismatch": 0, "reachable_mismatch": 0, "natext_max_err": 0.0, "cases": 0,
       "reachable_cases": 0}


def singleton_np(T, I):
    n = len(T)
    return {frozenset({i}): (T[i], I[i], 1 - T[i] - I[i]) for i in range(n)}


for trial in range(6000):
    n = int(rng.integers(2, 6))
    # draw (T_i, I_i, F_i) on the simplex, then shrink T to hit all regimes
    x = rng.dirichlet(np.ones(3) * rng.uniform(0.3, 3), size=n)
    scale = rng.uniform(0.2, 1.5)
    T = np.clip(x[:, 0] * scale, 0, 1)
    I = np.minimum(x[:, 1], 1 - T)
    NPs = singleton_np(T, I)
    d0 = 1 - T.sum()
    proper_formula = (d0 >= -1e-12) and (d0 <= I.sum() + 1e-12)
    proper_lp = credal_nonempty(NPs, n)
    out["proper_mismatch"] += proper_formula != proper_lp
    reach_formula = proper_formula and all(I[i] <= d0 + 1e-12 and d0 <= I.sum() - I[i] + 1e-12
                                           for i in range(n))
    reach_lp = is_coherent(NPs, n)
    out["reachable_mismatch"] += reach_formula != reach_lp
    if reach_lp:
        out["reachable_cases"] += 1
        for A in events(n):
            idx = list(A)
            nidx = list(complement(A, n))
            TA, IA, InA = T[idx].sum(), I[idx].sum(), I[nidx].sum()
            Lf = TA + max(d0 - InA, 0.0)
            Uf = TA + min(IA, d0)
            Llp = natural_extension(NPs, n, A)
            Ulp = natural_extension(NPs, n, A, upper=True)
            out["natext_max_err"] = max(out["natext_max_err"], abs(Lf - Llp), abs(Uf - Ulp))
    out["cases"] += 1

assert out["proper_mismatch"] == 0 and out["reachable_mismatch"] == 0
assert out["natext_max_err"] < 1e-7

# T5a: IDM with prior strength s <-> evidential NP <-> SL opinion ------------------------------
idm = {"cases": 0, "max_err": 0.0}
for _ in range(2000):
    k = int(rng.integers(2, 6))
    counts = rng.integers(0, 20, size=k).astype(float)
    s = float(rng.uniform(0.5, 4))
    N = counts.sum()
    T = counts / (N + s)
    I = np.full(k, s / (N + s))
    NPs = singleton_np(T, I)
    assert is_coherent(NPs, k)  # IDM singleton intervals are reachable
    for A in events(k):
        nA = counts[list(A)].sum()
        lo = natural_extension(NPs, k, A)
        hi = natural_extension(NPs, k, A, upper=True)
        idm["max_err"] = max(idm["max_err"], abs(lo - nA / (N + s)), abs(hi - (nA + s) / (N + s)))
    # binomial SL opinion with W = s: b = r/(r+q+W), u = W/(r+q+W)
    r, q = counts[0], counts[1:].sum()
    b, u = r / (r + q + s), s / (r + q + s)
    idm["max_err"] = max(idm["max_err"], abs(b - T[0]), abs(u - I[0]))
    idm["cases"] += 1
assert idm["max_err"] < 1e-7
out["idm_sl_identity"] = idm

save("t2_intervals_idm.json", out)
print("T2/T5a OK", out)
