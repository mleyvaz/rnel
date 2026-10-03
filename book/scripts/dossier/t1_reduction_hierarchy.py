"""T1 (reduction), T6 (belief functions) and the strict hierarchy, with counterexamples.

Checks
 1. round trip NP(normalized, dual) <-> conjugate pair (L, U = 1 - L(not A)), I = U - L;
 2. normalization + duality do NOT imply avoiding sure loss nor coherence (C1, C2);
 3. n = 2: every normalized dual NP is coherent;
 4. belief functions give normalized dual coherent NP with I(A) = straddling mass;
 5. strictness of Bel < 2-monotone < coherent (C3, C4) found by search and re-checked by LP.
"""
import itertools

import numpy as np

from common import (complement, events, is_coherent, credal_nonempty, natural_extension,
                    np_from_lower, lower_envelope, mobius, full_L, is_2monotone,
                    sure_loss_degree, save)

rng = np.random.default_rng(20260929)
out = {}


def random_dual_np(n):
    """Random normalized dual NP: draw T(A), T(not A) with T(A)+T(not A) <= 1."""
    L = {}
    for A in events(n):
        Ac = complement(A, n)
        if A in L:
            continue
        x = rng.dirichlet([1, 1, 1])  # (T(A), T(Ac), I)
        L[A], L[Ac] = x[0], x[1]
    return np_from_lower(L, n), L


# 1-2. round trip + classification ----------------------------------------------------------
stats = {}
for n in (2, 3, 4):
    counts = {"sure_loss": 0, "asl_incoherent": 0, "coherent": 0}
    roundtrip_err = 0.0
    N = 3000 if n < 4 else 1500
    for _ in range(N):
        NP, L = random_dual_np(n)
        for A, (T, I, F) in NP.items():
            Ac = complement(A, n)
            U = 1 - L[Ac]
            roundtrip_err = max(roundtrip_err, abs(T - L[A]), abs(I - (U - L[A])),
                                abs(T + I + F - 1), abs(NP[Ac][0] - F), abs(NP[Ac][2] - T),
                                abs(NP[Ac][1] - I))
        if not credal_nonempty(NP, n):
            counts["sure_loss"] += 1
            assert sure_loss_degree(NP, n) > 0
        elif is_coherent(NP, n):
            counts["coherent"] += 1
        else:
            counts["asl_incoherent"] += 1
            # natural extension dominates T and corrects it somewhere
            assert all(natural_extension(NP, n, A) >= T - 1e-9 for A, (T, _, _) in NP.items())
    stats[n] = {"samples": N, **counts, "max_roundtrip_error": roundtrip_err}
    if n == 2:
        assert counts["coherent"] == N, "n=2: all normalized dual NP must be coherent"
out["random_normalized_dual"] = stats

# Counterexample C1: normalized, dual, T(A)+T(not A) <= 1, but sure loss (n = 3)
n = 3
L1 = {}
for A in events(n):
    L1[A] = 0.4 if len(A) == 1 else 0.3  # singletons 0.4; pairs 0.3 = F of singleton
C1 = np_from_lower(L1, n)
out["C1"] = {"NP": {str(sorted(A)): v for A, v in C1.items()},
             "nonempty": credal_nonempty(C1, n), "delta": sure_loss_degree(C1, n)}
assert not credal_nonempty(C1, n)

# Counterexample C2: avoids sure loss, incoherent (non-monotone: T({0,1}) < T({0}))
L2 = {frozenset({0}): 0.5, frozenset({1}): 0.0, frozenset({2}): 0.0,
      frozenset({0, 1}): 0.2, frozenset({0, 2}): 0.5, frozenset({1, 2}): 0.0}
C2 = np_from_lower(L2, n)
ne = natural_extension(C2, n, frozenset({0, 1}))
out["C2"] = {"NP": {str(sorted(A)): v for A, v in C2.items()},
             "nonempty": credal_nonempty(C2, n), "coherent": is_coherent(C2, n),
             "T({0,1})": 0.2, "natural_extension({0,1})": ne}
assert credal_nonempty(C2, n) and not is_coherent(C2, n) and abs(ne - 0.5) < 1e-9

# 4. belief functions ------------------------------------------------------------------------
bel_ok = 0
for n in (2, 3, 4):
    allev = events(n, proper=False)
    focal = [A for A in allev if A]
    for _ in range(500):
        w = rng.dirichlet(np.ones(len(focal)) * 0.5)
        m = dict(zip(focal, w))
        Bel = {A: sum(m[B] for B in focal if B <= A) for A in events(n)}
        NP = np_from_lower(Bel, n)
        assert is_coherent(NP, n)
        for A, (T, I, F) in NP.items():
            strad = sum(m[B] for B in focal if (B & A) and not B <= A)
            assert abs(I - strad) < 1e-9
        bel_ok += 1
out["belief_functions_coherent_and_I_is_straddling_mass"] = bel_ok

# 5. strictness: coherent not 2-monotone (n=4) and 2-monotone not belief (n=3) ----------------
def random_envelope(n, k):
    d = rng.dirichlet(np.ones(n) * 0.7, size=k)
    return lower_envelope(d, n, events(n))

C3 = None
for _ in range(20000):
    L = random_envelope(4, 3)
    if not is_2monotone(L, 4):
        C3 = L
        break
assert C3 is not None and is_coherent(np_from_lower(C3, 4), 4)
C4 = None
n3_all_2mono = True
for _ in range(5000):
    L = random_envelope(3, 3)
    n3_all_2mono &= is_2monotone(L, 3)
    m = mobius(full_L(L, 3), 3)
    if C4 is None and min(m.values()) < -1e-6:
        C4 = (L, m)
assert C4 is not None and is_2monotone(C4[0], 3)
out["C3_coherent_not_2monotone_n4"] = {str(sorted(A)): v for A, v in C3.items()}
out["C4_2monotone_not_belief_n3"] = {
    "L": {str(sorted(A)): v for A, v in C4[0].items()},
    "mobius_min": min(C4[1].values())}
out["n3_all_sampled_envelopes_2monotone"] = n3_all_2mono

save("t1_reduction_hierarchy.json", out)
print("T1/T6 OK")
for k, v in stats.items():
    print(" n=%d" % k, v)
print(" C1 delta =", round(out["C1"]["delta"], 6))
print(" C2 T({0,1}) = 0.2 but natural extension =", ne)
print(" belief-function cases:", bel_ok, "; n=3 envelopes all 2-monotone:", n3_all_2mono)
