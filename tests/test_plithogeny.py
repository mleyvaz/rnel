"""Tests of rnel.plithogeny against the numbered results of Chapter 10 of the book (and the script p1_plithogenic.py)."""
import itertools

import numpy as np
import pytest

from rnel import copula as cop
from rnel import plithogeny as pl

COPULAS = {
    "Pi": (cop.pi_copula, True), "M": (cop.upper_frechet, True), "W": (cop.lower_frechet, True),
    "FGM(0.5)": (cop.fgm(0.5), True), "FGM(-1)": (cop.fgm(-1), True),
    "Frank(5)": (cop.frank(5.0), True), "Frank(-5)": (cop.frank(-5.0), True),
    "Clayton(2)": (cop.clayton(2.0), False), "Clayton(8)": (cop.clayton(8.0), False),
    "survClayton(2)": (cop.survival(cop.clayton(2.0)), False), "Gumbel(2)": (cop.gumbel(2.0), False),
}


def simplex(rng):
    e = rng.exponential(size=3)
    return e / e.sum()


# ------------------------------------------------------------------------------------------------ Theorem 10
@pytest.mark.theorem("Theorem 10(a)")
def test_theorem10a_lower_envelope_for_every_copula_is_attained_at_lower_endpoints():
    """For each copula the switched event probabilities increase in the marginals, so the lower envelope over
    the credal set is the plithogenic N-norm (grid search over p in [T1, 1], q in [T2, 1])."""
    rng = np.random.default_rng(20260930)
    grid = np.linspace(0, 1, 21)
    worst = 0.0
    for C, _ in COPULAS.values():
        for _ in range(60):
            x, y, c = rng.random(3), rng.random(3), rng.random()
            target = pl.plith_nnorm(x, y, c, C)
            vals = []
            for X, g in [(0, c), (1, 0.5), (2, 1 - c)]:
                ps, qs = x[X] + (1 - x[X]) * grid, y[X] + (1 - y[X]) * grid
                vals.append(min(pl.switched_lower(p, q, g, C) for p in ps for q in qs))
            worst = max(worst, np.abs(np.array(vals) - np.array(target)).max())
    assert worst < 1e-9


@pytest.mark.theorem("Theorem 10(a)")
def test_theorem10a_formula_equals_definition5_with_copula_coupling():
    rng = np.random.default_rng(1)
    for C, _ in COPULAS.values():
        for _ in range(50):
            x, y, c = rng.random(3), rng.random(3), rng.random()
            T, I, F = pl.plith_nnorm(x, y, c, C)
            assert T == pytest.approx(pl.switched_probability(x[0], y[0], C(x[0], y[0]), c, "conj"), abs=1e-12)
            assert F == pytest.approx(pl.switched_probability(x[2], y[2], C(x[2], y[2]), c, "disj"), abs=1e-12)
            assert I == pytest.approx(0.5 * (x[1] + y[1]), abs=1e-12)


@pytest.mark.theorem("Theorem 10(b)")
def test_theorem10b_no_dependence_closed_form_matches_lp_on_16_atoms():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(20260930)
    err = 0.0
    for k in range(400):
        x, y = rng.random(3), rng.random(3)
        c = rng.random() if k % 10 else [0.0, 0.5, 1.0][k % 3]
        err = max(err, np.abs(np.array(pl.plith_nnorm_lp(x, y, c)) - np.array(pl.plith_nnorm_nodep(x, y, c))).max())
    assert err < 1e-8


@pytest.mark.theorem("Theorem 10(b)", "Theorem 9(b)")
def test_theorem10b_at_c0_returns_theorem9b_on_truth_and_falsity():
    rng = np.random.default_rng(3)
    for _ in range(200):
        x, y = rng.random(3), rng.random(3)
        T, _, F = pl.plith_nnorm_nodep(x, y, 0.0)
        assert T == pytest.approx(max(0.0, x[0] + y[0] - 1))
        assert F == pytest.approx(max(x[2], y[2]))


@pytest.mark.theorem("Theorem 10(c)")
def test_theorem10c_half_switch_gives_the_mean_for_every_model_and_only_there():
    rng = np.random.default_rng(4)
    for C, _ in COPULAS.values():
        for _ in range(100):
            x, y = rng.random(3), rng.random(3)
            assert np.allclose(pl.plith_nnorm(x, y, 0.5, C), (x + y) / 2, atol=1e-12)
    x, y = rng.random(3), rng.random(3)
    assert np.allclose(pl.plith_nnorm_nodep(x, y, 0.5), (x + y) / 2)
    assert pl.is_dependence_free(0.5)
    for c in (0.0, 0.1, 0.3, 0.49, 0.51, 0.8, 1.0):
        assert not pl.is_dependence_free(c)


@pytest.mark.theorem("Theorem 10(d)")
def test_theorem10d_indeterminacy_does_not_depend_on_the_model():
    rng = np.random.default_rng(5)
    for _ in range(200):
        x, y, c = rng.random(3), rng.random(3), rng.random()
        Is = [pl.plith_nnorm(x, y, c, C)[1] for C, _ in COPULAS.values()] + [pl.plith_nnorm_nodep(x, y, c)[1]]
        assert max(Is) - min(Is) < 1e-12


# ------------------------------------------------------------------------------------------------ Theorem 11
@pytest.mark.theorem("Theorem 11(a)", "Theorem 11(b)", "Theorem 11(c)")
def test_theorem11_classical_frame_truth_exact_falsity_gap_and_surplus_formulas():
    rng = np.random.default_rng(6)
    for name, (C, radially_symmetric) in COPULAS.items():
        max_gap = 0.0
        for _ in range(800):
            x, y, c = simplex(rng), simplex(rng), rng.random()
            r = pl.classical_frame(x, y, c, C)
            assert abs(r["truth_gap"]) < 1e-12
            assert abs(r["falsity_gap"] - r["falsity_gap_formula"]) < 1e-9
            assert abs(r["surplus"] - r["surplus_formula"]) < 1e-9
            max_gap = max(max_gap, abs(r["falsity_gap"]))
        if radially_symmetric:
            assert max_gap < 1e-9, name
        else:
            assert max_gap > 1e-4, name


@pytest.mark.theorem("Theorem 11(b)")
def test_theorem11b_falsity_exact_at_half_for_non_symmetric_copula():
    rng = np.random.default_rng(7)
    C = cop.clayton(2.0)
    for _ in range(200):
        x, y = simplex(rng), simplex(rng)
        assert abs(pl.classical_frame(x, y, 0.5, C)["falsity_gap"]) < 1e-12


# ------------------------------------------------------------------------------------------------ Corollary 3
@pytest.mark.theorem("Corollary 3(a)", "Corollary 3(b)")
def test_corollary3_minimal_lifting_atoms_and_faithfulness_for_hybrid_types():
    pytest.importorskip("scipy")
    assert len(pl.minimal_lifting([3, 3, 3])) == 10
    assert pl.lifting_atom_counts([1, 2, 3]) == (7, 24)
    rng = np.random.default_rng(8)
    for ms in ([3, 3, 3], [1, 2, 3], [3]):
        for _ in range(25):
            assert pl.is_faithful(rng.random(sum(ms)), ms)


@pytest.mark.theorem("Corollary 3(b)")
def test_corollary3b_m_atoms_are_not_enough_on_the_whole_cube():
    """With only the m atoms that drop one event each (no all-ones atom) the triple (1, 1, 1) is not representable."""
    pytest.importorskip("scipy")
    from scipy.optimize import linprog
    atoms = pl.minimal_lifting(3)[:-1]
    A = np.array(atoms, dtype=float).T
    res = linprog(np.zeros(3), A_ub=-A, b_ub=-np.ones(3), A_eq=[np.ones(3)], b_eq=[1], bounds=(0, 1), method="highs")
    assert res.status == 2  # infeasible


@pytest.mark.theorem("Corollary 3(c)")
def test_corollary3c_product_lifting_is_faithful_but_larger():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(9)
    ms = [2, 3]
    minimal, product = pl.lifting_atom_counts(ms)
    assert (minimal, product) == (6, 12) and product > minimal
    assert len(pl.product_lifting(ms)) == 12
    for _ in range(20):
        assert pl.is_faithful(rng.random(5), ms, product=True)


# ------------------------------------------------------------------------------------------------ Corollary 4
@pytest.mark.theorem("Corollary 4")
def test_corollary4_plithogenic_idm_coherent_and_natural_extension():
    pytest.importorskip("scipy")
    from scipy.optimize import linprog
    rng = np.random.default_rng(10)
    for _ in range(300):
        k = int(rng.integers(2, 6))
        n = rng.integers(0, 30, size=k)
        nq = int(rng.integers(0, 20))
        s = float(rng.uniform(0.5, 4))
        trip = np.array(pl.plithogenic_idm(n, nq, s))
        T, I = trip[:, 0], trip[:, 1]
        D = 1 - T.sum()
        assert -1e-12 <= D <= I.sum() + 1e-12  # proper
        assert all(I[i] <= D + 1e-12 and D <= I.sum() - I[i] + 1e-12 for i in range(k))  # reachable
        A = sorted(set(rng.choice(k, size=int(rng.integers(1, k)), replace=False).tolist()))
        cvec = np.array([1.0 if i in A else 0.0 for i in range(k)])
        bounds = list(zip(T, T + I))
        lo = linprog(cvec, A_eq=[np.ones(k)], b_eq=[1], bounds=bounds, method="highs").fun
        hi = -linprog(-cvec, A_eq=[np.ones(k)], b_eq=[1], bounds=bounds, method="highs").fun
        tA, iA, fA = pl.plithogenic_idm(n, nq, s, event=A)
        assert lo == pytest.approx(tA, abs=1e-9) and hi == pytest.approx(tA + iA, abs=1e-9)


@pytest.mark.theorem("Corollary 4", "Table 7")
def test_corollary4_table7_north_region():
    T, I, F = pl.plithogenic_idm([112, 38], 20, 2.0)[0]
    assert (round(T, 4), round(I, 4), round(F, 4)) == (0.6512, 0.1279, 0.2209)


# ------------------------------------------------------------------------------------------------ Corollary 5
@pytest.mark.theorem("Corollary 5")
def test_corollary5_three_conjunctions_lp_and_only_half_is_dependence_free():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(11)
    spread = {0.0: 0.0, 1.0: 0.0, 0.5: 0.0}
    for _ in range(200):
        x, y = rng.random(3), rng.random(3)
        for kind, g in pl.SMARANDACHE_CONJUNCTIONS.items():
            lp = pl.plith_nnorm_lp(x, y, 0.0, gamma_I=g)[1]
            form = pl.smarandache_conjunction(x, y, kind)[1]
            assert lp == pytest.approx(form, abs=1e-8)
            vals = [pl.smarandache_conjunction(x, y, kind, C)[1] for C in (cop.lower_frechet, cop.pi_copula, cop.upper_frechet)]
            spread[g] = max(spread[g], max(vals) - min(vals))
        I1, I2 = x[1], y[1]
        assert pl.smarandache_conjunction(x, y, "eq171")[1] == pytest.approx(max(0.0, I1 + I2 - 1))
        assert pl.smarandache_conjunction(x, y, "eq174")[1] == pytest.approx(max(I1, I2))
        assert pl.smarandache_conjunction(x, y, "plithogenic")[1] == pytest.approx((I1 + I2) / 2)
    assert spread[0.5] < 1e-12 and spread[0.0] > 0.1 and spread[1.0] > 0.1


# ------------------------------------------------------------------------------------------------ Proposition 2
@pytest.mark.theorem("Proposition 2(a)")
def test_proposition2a_commutative_not_associative_defect_formula():
    rng = np.random.default_rng(12)
    for a, b, d, c in rng.random((5000, 4)):
        assert pl.switch_truth(a, b, c) == pytest.approx(pl.switch_truth(b, a, c))
        assert pl.associativity_defect(a, b, d, c) == pytest.approx(c * (1 - c) * (d - a), abs=1e-12)
    for C in (cop.upper_frechet, cop.lower_frechet):
        worst = max(abs(pl.associativity_defect(a, b, d, c, C)) for a, b, d, c in rng.random((5000, 4)))
        assert worst > 1e-3


@pytest.mark.theorem("Proposition 2(b)")
def test_proposition2b_nary_forms_are_lower_envelopes_and_reduce_to_theorem10():
    rng = np.random.default_rng(13)
    for _ in range(500):
        n = int(rng.integers(3, 6))
        x, c = rng.random(n), rng.random()
        ind, com = pl.nary_switched(x, c), pl.nary_switched(x, c, "comonotone")
        for _ in range(10):
            p = x + (1 - x) * rng.random(n)
            assert pl.nary_switched(p, c) >= ind - 1e-12
            assert pl.nary_switched(p, c, "comonotone") >= com - 1e-12
        perm = rng.permutation(n)
        assert pl.nary_switched(x[perm], c) == pytest.approx(ind)
    a, b, c = 0.3, 0.7, 0.2
    assert pl.nary_switched([a, b], c) == pytest.approx(pl.switch_truth(a, b, c, cop.pi_copula))
    assert pl.nary_switched([a, b], c, "comonotone") == pytest.approx(pl.switch_truth(a, b, c, cop.upper_frechet))


@pytest.mark.theorem("Proposition 2(c)")
def test_proposition2c_half_switch_not_dependence_free_for_n3_mean_is():
    pytest.importorskip("scipy")
    x = [0.3, 0.6, 0.8]
    assert round(pl.nary_switched(x, 0.5), 4) == 0.544
    assert round(pl.nary_switched(x, 0.5, "comonotone"), 4) == 0.55
    assert round(pl.selection_mean(x), 4) == 0.5667
    rng = np.random.default_rng(14)
    spread = 0.0
    for _ in range(100):
        p = rng.random(int(rng.integers(3, 6)))
        lo, hi = pl.nary_switched_lp(p, 0.5)
        spread = max(spread, hi - lo)
    assert spread > 0.05
    assert pl.selection_mean([0.2, 0.9]) == pytest.approx(pl.nary_switched([0.2, 0.9], 0.5))


# ------------------------------------------------------------------------------------------------ applications
@pytest.mark.theorem("Section 11 (plithogenic cognitive map)")
def test_cognitive_map_step_and_union_lower_bound():
    pytest.importorskip("scipy")
    from scipy.optimize import linprog
    assert round(pl.pcm_step(1.0, 0.5, 0.2), 4) == 0.6
    assert pl.pcm_step(0.0, 0.5, 0.2) == pytest.approx(0.2 * 0.5)
    rng = np.random.default_rng(15)
    for _ in range(50):
        ps = rng.random(int(rng.integers(2, 5)))
        pats = list(itertools.product([0, 1], repeat=len(ps)))
        obj = [float(any(s)) for s in pats]
        Aeq = [[s[i] for s in pats] for i in range(len(ps))] + [[1] * len(pats)]
        lo = linprog(obj, A_eq=Aeq, b_eq=list(ps) + [1], bounds=(0, 1), method="highs").fun
        assert lo == pytest.approx(pl.pcm_aggregate(ps), abs=1e-9)


@pytest.mark.theorem("Section 11 (application A3)")
def test_application_a3_contradiction_pool():
    V = [(0.62, 0.30, 0.08), (0.55, 0.20, 0.35), (0.70, 0.10, 0.60)]
    cw = pl.contradiction_pool(V, [0.0, 1 / 3, 2 / 3])
    w = np.array([1, 2 / 3, 1 / 3]) / 2
    assert np.allclose(cw, w @ np.array(V))
    assert np.allclose(pl.contradiction_pool(V, [0.2] * 3), np.mean(V, axis=0))
