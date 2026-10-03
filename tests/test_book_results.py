"""Book results of Chapters 3-7 and 12 that had no explicit test before 0.3.0 (Corollary 1, Theorem 3 cases,
Theorem 4, Proposition 3, Section 12.3 numbers)."""
import itertools

import numpy as np
import pytest

pytest.importorskip("scipy")

from rnel import neutro_credal as nc

# lower probabilities of the counterexamples C3 and C4 (results file of t1_reduction_hierarchy.py, Appendix B)
C3 = {(0,): 0.09229276032189059, (1,): 0.005206460242533165, (2,): 0.03439367634300739, (3,): 0.01224636160535773,
      (0, 1): 0.28816555423640977, (0, 2): 0.2842416885496426, (0, 3): 0.10453912192724832,
      (1, 2): 0.4045677922649048, (1, 3): 0.2479976631112588, (2, 3): 0.27331681161957216,
      (0, 1, 2): 0.65441580447154, (0, 1, 3): 0.5309567571051355, (0, 2, 3): 0.36560957194146276,
      (1, 2, 3): 0.7170409060061234}
C4 = {(0,): 0.13975133220199656, (1,): 0.026200355140305494, (2,): 0.17955975838817476,
      (0, 1): 0.2453026360838626, (0, 2): 0.8944486961181338, (1, 2): 0.23019444001172373}


def full_lower(L, n):
    out = dict(L)
    out[()] = 0.0
    out[tuple(range(n))] = 1.0
    return out


def to_np(L, n):
    """NP(A) = (L(A), U(A) - L(A), 1 - U(A)) with U(A) = 1 - L(complement of A)."""
    Lf = full_lower(L, n)
    np_events = {}
    for A, l in L.items():
        comp = tuple(i for i in range(n) if i not in A)
        u = 1 - Lf[comp]
        np_events[A] = (l, u - l, 1 - u)
    return np_events


def two_monotone(L, n, tol=1e-12):
    Lf = full_lower(L, n)
    ev = list(Lf)
    for A in ev:
        for B in ev:
            U = tuple(sorted(set(A) | set(B)))
            I = tuple(sorted(set(A) & set(B)))
            if Lf[U] + Lf[I] < Lf[A] + Lf[B] - tol:
                return False
    return True


def mobius_min(L, n):
    Lf = full_lower(L, n)
    m = {}
    for A in sorted(Lf, key=len):
        m[A] = Lf[A] - sum(m[B] for B in m if set(B) < set(A))
    return min(m.values())


@pytest.mark.theorem("Corollary 1")
def test_corollary1_strict_hierarchy_separating_examples():
    # C4: 2-monotone, not a belief function (negative Möbius value -0.024)
    assert two_monotone(C4, 3)
    assert round(mobius_min(C4, 3), 3) == -0.024
    # C3: coherent, not 2-monotone (n = 4)
    assert nc.is_coherent(to_np(C3, 4), outcomes=range(4), method="lp")
    assert not two_monotone(C3, 4)
    # C2: avoids sure loss, incoherent; C1: sure loss (Section 3 counterexamples)
    C2 = {(0,): (0.5, 0.5, 0.0), (1,): (0.0, 0.5, 0.5), (2,): (0.0, 0.8, 0.2), (0, 1): (0.2, 0.8, 0.0),
          (0, 2): (0.5, 0.5, 0.0), (1, 2): (0.0, 0.5, 0.5)}
    assert nc.classify(C2, outcomes=range(3)) == "incoherent"
    C1 = {e: ((0.4, 0.3, 0.3) if len(e) == 1 else (0.3, 0.3, 0.4)) for e in
          [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]}
    assert nc.classify(C1, outcomes=range(3)) == "sure_loss"
    assert nc.sure_loss_degree(C1, outcomes=range(3)) == pytest.approx(1 / 15, abs=1e-9)


@pytest.mark.theorem("Corollary 1")
def test_corollary1_belief_functions_are_coherent_and_I_is_the_straddling_mass():
    rng = np.random.default_rng(0)
    n = 4
    subsets = [s for r in range(1, n + 1) for s in itertools.combinations(range(n), r)]
    for _ in range(50):
        focal = rng.choice(len(subsets), size=4, replace=False)
        w = rng.dirichlet(np.ones(4))
        m = {subsets[k]: wk for k, wk in zip(focal, w)}
        events = [s for s in subsets if len(s) < n]
        L = {A: sum(v for B, v in m.items() if set(B) <= set(A)) for A in events}
        NP = to_np(L, n)
        assert nc.is_coherent(NP, outcomes=range(n), method="lp")
        for A in events:
            straddle = sum(v for B, v in m.items() if set(B) & set(A) and not set(B) <= set(A))
            assert NP[A][1] == pytest.approx(straddle, abs=1e-12)


@pytest.mark.theorem("Theorem 3")
def test_theorem3_sure_loss_degree_including_off_values():
    assert nc.sure_loss_degree((0.6, 0.1, 0.6)) == pytest.approx(0.1)
    assert nc.sure_loss_degree((0.3, 0.9, 0.5)) == 0.0          # I plays no role
    assert nc.sure_loss_degree((1.3, 0.0, 0.0)) == pytest.approx(0.3)   # T > 1
    assert nc.sure_loss_degree((-0.2, 0.0, 0.5)) == 0.0               # T < 0 does not create sure loss
    assert nc.sure_loss_degree((0.2, 0.0, 1.4)) == pytest.approx(0.4)   # F > 1


@pytest.mark.theorem("Theorem 4(a)", "Theorem 4(b)")
def test_theorem4_only_admissible_affine_reading_is_walley_dempster():
    """Readings l = T + alpha s, u = T + I + beta s with s = T + I + F - 1: only (alpha, beta) = (0, -1) keeps
    0 <= l, u <= 1 on the whole cube; its kernel is the I direction."""
    corners = list(itertools.product([0.0, 1.0], repeat=3))
    admissible = []
    for alpha in np.linspace(-2, 2, 81):
        for beta in np.linspace(-2, 2, 81):
            ok = True
            for T, I, F in corners:  # affine, so the corners suffice
                s = T + I + F - 1
                l, u = T + alpha * s, T + I + beta * s
                if not (-1e-12 <= l <= 1 + 1e-12 and -1e-12 <= u <= 1 + 1e-12):
                    ok = False
                    break
            if ok:
                admissible.append((round(alpha, 6), round(beta, 6)))
    assert admissible == [(0.0, -1.0)]
    # linear part of (l, u) = (T, 1 - F): kernel spanned by (0, 1, 0)
    A = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    _, _, vt = np.linalg.svd(A)
    k = vt[-1]
    assert np.allclose(np.abs(k), [0, 1, 0])


@pytest.mark.theorem("Proposition 3(c)")
def test_proposition3c_hull_of_two_experts_is_not_event_determined():
    from scipy.optimize import linprog
    P1, P2 = np.array([0.2, 0.3, 0.5]), np.array([0.4, 0.5, 0.1])
    events = [e for r in (1, 2) for e in itertools.combinations(range(3), r)]
    low = {e: min(P1[list(e)].sum(), P2[list(e)].sum()) for e in events}
    assert [round(low[e], 4) for e in events] == [0.2, 0.3, 0.1, 0.5, 0.5, 0.6]
    A_ub = [[-1.0 if i in e else 0.0 for i in range(3)] for e in events]
    b_ub = [-low[e] for e in events]
    f = np.array([0.0, 100.0, 10.0])
    core = [linprog(s * f, A_ub=A_ub, b_ub=b_ub, A_eq=[[1, 1, 1]], b_eq=[1], bounds=(0, 1), method="highs").fun * s
            for s in (1, -1)]
    assert [round(v, 6) for v in core] == [33.0, 53.0]
    assert sorted([P1 @ f, P2 @ f]) == [35.0, 51.0]


@pytest.mark.theorem("Section 12.3 (credal classifier, Manski bounds)")
def test_section_12_3_idm_and_manski_bounds():
    ns, nn, nq = 112, 38, 20
    N = ns + nn + nq
    assert (round(ns / N, 4), round((ns + nq) / N, 4)) == (0.6588, 0.7765)
    lo, hi = ns / (N + 2), (ns + nq + 2) / (N + 2)
    assert (round(lo, 4), round(hi, 4)) == (0.6512, 0.7791)
    # credal classifier with the IDM (s = 2): counts (9, 7, 2) leave classes 1 and 2 undominated
    n = np.array([9, 7, 2]); s = 2; Nn = n.sum()
    lo, hi = n / (Nn + s), (n + s) / (Nn + s)
    und = [k + 1 for k in range(3) if not any(lo[j] > hi[k] for j in range(3) if j != k)]
    assert und == [1, 2]


# ================================================================================================ Part VI
@pytest.mark.theorem("Proposition 16.1")
def test_proposition16_1_tuple_converges_to_frequencies():
    from rnel import Reports, rnel_tuple
    tp = rnel_tuple(Reports(t=30, f=10), W=2.0)
    assert (round(tp.T, 3), round(tp.F, 3), round(tp.G, 3)) == (0.714, 0.238, 0.048)
    big = rnel_tuple(Reports(t=30000, f=10000), W=2.0)
    assert round(big.T, 3) == 0.750 and round(big.G, 3) == 0.000


@pytest.mark.theorem("Proposition 16.4", "Corollary 2")
def test_proposition16_4_idm_is_the_sl_opinion_from_evidence():
    from rnel.sl import Opinion
    T, I, F = nc.idm_triple([6, 2], s=2.0)[0]
    assert (T, I, F) == pytest.approx((0.6, 0.2, 0.2))
    op = Opinion.from_evidence(6, 2, 2.0)
    assert (op.b, op.u, op.d) == pytest.approx((T, I, F))


@pytest.mark.theorem("Proposition 17.1")
def test_proposition17_1_typed_head_contains_evidential_deep_learning():
    torch = pytest.importorskip("torch")
    from rnel.nn import rnel_from_evidence
    out = rnel_from_evidence(torch.tensor([3.0, 1.0, 0.0, 0.0, 0.0]), W=2.0)
    T, F, C, U, N, G = out.tolist()
    assert (round(T, 3), round(F, 3), round(G, 3)) == (0.5, 0.167, 0.333) and C == U == N == 0
    alpha = torch.tensor([4.0, 2.0])  # Sensoy et al. (2018): b_k = e_k / S, u = K / S with S = sum alpha
    S = alpha.sum()
    assert T == pytest.approx(float((alpha[0] - 1) / S)) and G == pytest.approx(float(2 / S))
    g = torch.Generator().manual_seed(0)
    for K in (2, 3, 6):  # K classes, W = K
        e = torch.rand(50, K, generator=g) * 10
        S = e.sum(-1, keepdim=True) + K
        full = torch.cat([e, torch.zeros(50, 5 - K)], -1) if K <= 5 else None
        if full is not None:
            out = rnel_from_evidence(full, W=float(K))
            assert torch.allclose(out[:, :K], e / S) and torch.allclose(out[:, -1:], K / S)


@pytest.mark.theorem("Proposition 17.2")
def test_proposition17_2_dissonance_decomposes_into_between_and_within():
    from rnel import conflict as cf
    W = 2.0
    rng = np.random.default_rng(1)
    for _ in range(300):
        prof = [tuple(map(float, rng.integers(0, 12, size=2))) for _ in range(int(rng.integers(1, 6)))]
        R, S = sum(r for r, _ in prof), sum(s for _, s in prof)
        diss = 2 * min(R, S) / (R + S + W)
        assert diss == pytest.approx(2 * cf.k_between(prof) / (R + S + W) + 2 * cf.k_within(prof) / (R + S + W))
    dispute, balanced = [(8, 0), (0, 8)], [(8, 8)]
    assert round(cf.c_star(dispute, W), 3) == 0.889 and cf.c_star(balanced, W) == 0
    assert round(2 * cf.k_within(balanced) / 18, 3) == 0.889


@pytest.mark.theorem("Proposition 13A.4")
def test_proposition13A_4_dempster_of_one_sided_sources_is_cumulative_fusion():
    from rnel import Reports
    from rnel.dsmt import binary_frame, dempster, reports_to_bba
    sh = binary_frame("shafer")
    for W in (1.0, 2.0, 5.0):
        for r in range(0, 31, 5):
            for s in range(0, 31, 5):
                if r + s == 0:
                    continue
                mA = reports_to_bba(Reports(t=r), W=W, frame=sh)
                mB = reports_to_bba(Reports(f=s), W=W, frame=sh)
                pooled = reports_to_bba(Reports(t=r, f=s), W=W, frame=sh)
                d = dempster(sh, [mA, mB])
                assert all(d.get(k, 0.0) == pytest.approx(v, abs=1e-12) for k, v in pooled.items())


@pytest.mark.theorem("Proposition 13A.5")
def test_proposition13A_5_pcr6_needs_the_partial_conflicts():
    from rnel.dsmt import binary_frame, dsmc, pcr6
    sh, fr = binary_frame("shafer"), binary_frame("free")

    def bba(frame, x, nx, th):  # the book prints the second pair to six decimals: renormalise
        z = x + nx + th
        return {frame.element("x"): x / z, frame.element("not_x"): nx / z, frame.theta: th / z}
    a = [bba(sh, 0.6, 0.1, 0.3), bba(sh, 0.2, 0.5, 0.3)]
    b = [bba(sh, 0.65, 0.146027, 0.203973), bba(sh, 0.085714, 0.473051, 0.441234)]
    da = dsmc(fr, [bba(fr, *(m[sh.element("x")], m[sh.element("not_x")], m[sh.theta])) for m in a])
    db = dsmc(fr, [bba(fr, *(m[sh.element("x")], m[sh.element("not_x")], m[sh.theta])) for m in b])
    assert all(da.get(k, 0.0) == pytest.approx(db.get(k, 0.0), abs=2e-6) for k in set(da) | set(db))
    assert da.get(fr.element("x&not_x"), 0.0) == pytest.approx(0.32)
    xa, xb = pcr6(sh, a)[sh.element("x")], pcr6(sh, b)[sh.element("x")]
    assert xa == pytest.approx(0.53697, abs=5e-6) and xb == pytest.approx(0.542595, abs=5e-6)
