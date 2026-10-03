"""Tests of rnel.scores: numerical mirrors of the ten Lean 4 theorems on base rates and score functions
(formal/lean_scores/LeanScores/Basic.lean = formal/NeutroEvidence/NeutroEvidence/BaseRate.lean)."""
import numpy as np
import pytest

from rnel import scores as sc

rng = np.random.default_rng(20261003)
CUBE = rng.random((2000, 3))
NORM = rng.dirichlet([1, 1, 1], size=2000)


@pytest.mark.theorem("Base rates: classic_decomp (Lean)")
def test_classic_decomposition():
    for T, I, F in CUBE:
        assert sc.classic_score(T, I, F) == pytest.approx(2 / 3 * sc.S(0.5, T, F) + (1 - I) / 3)


@pytest.mark.theorem("Base rates: classic_eq_Sl_half_half (Lean)")
def test_classic_is_affine_in_S_half_half():
    for T, I, F in CUBE:
        assert sc.classic_score(T, I, F) == pytest.approx(2 / 3 * sc.S_lambda(0.5, 0.5, T, I, F) + 1 / 3)


@pytest.mark.theorem("Base rates: classic_normalized (Lean)")
def test_classic_normalised_case():
    for T, I, F in NORM:
        assert sc.classic_score(T, I, F) == pytest.approx((1 + 2 * T) / 3)


@pytest.mark.theorem("Base rates: classic_ranks_by_T (Lean)")
def test_classic_ranks_by_T_in_the_normalised_case():
    for x, y in zip(NORM[:1000], NORM[1000:]):
        assert (sc.classic_score(*x) < sc.classic_score(*y)) == (x[0] < y[0])


@pytest.mark.theorem("Base rates: accuracy_ranks_as_S_half (Lean)")
def test_accuracy_ranks_as_S_half():
    for x, y in zip(CUBE[:1000], CUBE[1000:]):
        assert (sc.accuracy(x[0], x[2]) < sc.accuracy(y[0], y[2])) == (sc.S(0.5, x[0], x[2]) < sc.S(0.5, y[0], y[2]))


@pytest.mark.theorem("Base rates: S_glut_form (Lean)")
def test_single_formula():
    for (T, I, F), a in zip(CUBE, rng.random(2000)):
        assert sc.S(a, T, F) == pytest.approx(T - a * (T + F - 1))


@pytest.mark.theorem("Base rates: S_gap_bounds (Lean)", "Base rates: S_glut_bounds (Lean)")
def test_gap_and_glut_bounds_and_hurwicz_reading():
    for (T, I, F), a in zip(CUBE, rng.random(2000)):
        s = sc.S(a, T, F)
        if T + F <= 1:
            assert T - 1e-12 <= s <= 1 - F + 1e-12 and sc.reading(T, F) in ("gap", "exact")
        else:
            assert 1 - F - 1e-12 <= s <= T + 1e-12 and sc.reading(T, F) == "glut"
        lo, hi = sc.S_bounds(T, F)
        assert lo - 1e-12 <= s <= hi + 1e-12
        assert s == pytest.approx(sc.hurwicz(T, 1 - F, a))


@pytest.mark.theorem("Base rates: Sl_normalized (Lean)", "Theorem 5")
def test_normalised_case_is_sl_projected_probability():
    for (T, I, F), a, lam in zip(NORM, rng.random(2000), rng.random(2000)):
        assert sc.S_lambda(a, lam, T, I, F) == pytest.approx(T + (a - lam) * I)
        assert sc.S(a, T, F) == pytest.approx(sc.projected_probability(T, I, a))


@pytest.mark.theorem("Base rates: dominance_interval (Lean)")
def test_dominance_at_endpoints_implies_dominance_on_the_interval():
    for _ in range(1000):
        x, y = rng.random(3), rng.random(3)
        aL, aU = np.sort(rng.random(2))
        if sc.dominates(x, y, (aL, aU)):
            for a in np.linspace(aL, aU, 25):
                assert sc.S(a, x[0], x[2]) >= sc.S(a, y[0], y[2]) - 1e-12
        lo, hi = sc.score_interval(x, (aL, aU))
        for a in np.linspace(aL, aU, 7):
            assert lo - 1e-12 <= sc.S(a, x[0], x[2]) <= hi + 1e-12


def test_partial_order_and_abstention_under_imprecise_base_rate():
    # x is better if the base rate is low (higher T), y if it is high (lower F)
    alts = {"x": (0.6, 0.3, 0.1), "y": (0.4, 0.6, 0.0), "z": (0.3, 0.3, 0.4)}
    assert sc.decide(alts, 0.2)[0] == "x"
    assert sc.decide(alts, 0.9)[0] == "y"
    choice, und = sc.decide(alts, (0.2, 0.9))
    assert choice is None and set(und) == {"x", "y"}
    assert ("x", "z") in sc.dominance_pairs(alts, (0.0, 1.0)) and ("y", "z") in sc.dominance_pairs(alts, (0.0, 1.0))
    with pytest.raises(ValueError):
        sc.S_lambda(0.5, 0, 0.5, 0.2, 0.3) and sc.dominates(alts["x"], alts["y"], (0.9, 0.2))
