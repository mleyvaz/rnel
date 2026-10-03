"""Fused base rate in cumulative fusion (Josang 2016, ch. 12)."""
import math
import random

from rnel.sl import Opinion, averaging_fusion, cumulative_fusion


def test_equal_base_rates_unchanged():
    A = Opinion.from_evidence(8, 2, a=0.3)
    B = Opinion.from_evidence(1, 9, a=0.3)
    assert math.isclose(cumulative_fusion(A, B).a, 0.3)


def test_equal_uncertainty_gives_mean():
    A = Opinion(0.5, 0.2, 0.3, 0.2)
    B = Opinion(0.1, 0.6, 0.3, 0.8)
    assert math.isclose(cumulative_fusion(A, B).a, 0.5)


def test_vacuous_source_is_neutral():
    A = Opinion(0.0, 0.0, 1.0, 0.9)      # vacuous, base rate 0.9
    B = Opinion(0.4, 0.3, 0.3, 0.2)
    F = cumulative_fusion(A, B)
    assert math.isclose(F.a, 0.2) and math.isclose(F.b, B.b) and math.isclose(F.u, B.u)


def test_formula_and_bounds_random():
    rng = random.Random(0)
    for _ in range(2000):
        ua, ub = rng.uniform(0.01, 0.99), rng.uniform(0.01, 0.99)
        ba, bb = rng.uniform(0, 1 - ua), rng.uniform(0, 1 - ub)
        aa, ab = rng.random(), rng.random()
        A, B = Opinion(ba, 1 - ua - ba, ua, aa), Opinion(bb, 1 - ub - bb, ub, ab)
        a = cumulative_fusion(A, B).a
        expected = (aa * ub + ab * ua - (aa + ab) * ua * ub) / (ua + ub - 2 * ua * ub)
        assert math.isclose(a, expected, rel_tol=1e-12)
        assert min(aa, ab) - 1e-12 <= a <= max(aa, ab) + 1e-12


def test_averaging_keeps_mean_base_rate():
    A = Opinion(0.5, 0.2, 0.3, 0.2)
    B = Opinion(0.1, 0.6, 0.3, 0.8)
    assert math.isclose(averaging_fusion(A, B).a, 0.5)
