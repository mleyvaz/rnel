"""Base-rate-calibrated conjunction and disjunction (Section 5.2 of Smarandache and Leyva-Vázquez)."""
import random

import pytest

from rnel import operators as op
from rnel import sl as SL
from rnel.sl import Opinion


def _norm(keys, rng):
    v = [rng.random() for _ in keys]
    s = sum(v)
    return {k: x / s for k, x in zip(keys, v)}


def _sl(x, a):
    b, u, d = op.coarsen(x)
    return Opinion(b, d, u, a)


FAMILIES = [
    (["T", "I", "N", "F"], op.tinf_and, op.tinf_or, op.tinf_not),
    (["T", "I", "N", "U1", "U2", "U3", "F"], op.mu_and, op.mu_or, op.mu_not),
    (["T", "C", "U", "N", "G", "F"], op.rnel_and, op.rnel_or, op.rnel_not),
]


@pytest.mark.parametrize("keys,conj,disj,neg", FAMILIES)
def test_calibrated_equals_sl_for_any_base_rates(keys, conj, disj, neg):
    rng = random.Random(7)
    for _ in range(2000):
        x, y = _norm(keys, rng), _norm(keys, rng)
        ax, ay = rng.random(), rng.random()
        m = SL.multiply(_sl(x, ax), _sl(y, ay))
        c = SL.comultiply(_sl(x, ax), _sl(y, ay))
        za, zo = op.calibrated_and(x, y, ax, ay, conj), op.calibrated_or(x, y, ax, ay, disj)
        assert op.coarsen(za) == pytest.approx((m.b, m.u, m.d), abs=1e-12)
        assert op.coarsen(zo) == pytest.approx((c.b, c.u, c.d), abs=1e-12)
        assert min(za.values()) >= -1e-12 and min(zo.values()) >= -1e-12
        assert sum(za.values()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("keys,conj,disj,neg", FAMILIES)
def test_calibrated_de_morgan_with_complemented_base_rates(keys, conj, disj, neg):
    rng = random.Random(11)
    for _ in range(1000):
        x, y = _norm(keys, rng), _norm(keys, rng)
        ax, ay = rng.random(), rng.random()
        lhs = neg(op.calibrated_and(x, y, ax, ay, conj))
        rhs = op.calibrated_or(neg(x), neg(y), 1 - ax, 1 - ay, disj)
        for k in keys:
            assert lhs[k] == pytest.approx(rhs[k], abs=1e-12)


def test_zero_base_rates_give_uncalibrated_conjunction():
    rng = random.Random(3)
    x, y = _norm(["T", "I", "N", "F"], rng), _norm(["T", "I", "N", "F"], rng)
    assert op.calibrated_and(x, y, 0.0, 0.0) == pytest.approx(op.tinf_and(x, y))
    assert op.calibrated_or(x, y, 1.0, 1.0) == pytest.approx(op.tinf_or(x, y))
