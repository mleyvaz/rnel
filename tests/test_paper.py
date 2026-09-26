"""The theorems and worked numbers of the SL/RNEL paper, as tests."""
import random

import pytest

from rnel import Opinion, Reports, fused_contradiction, rnel_tuple, sl_opinion
from rnel import operators as op
from rnel import sl
from rnel.ranking import mu_key, tinf_key

TOL = 1e-12


def rand_tuple(keys, normalized=True, rng=random):
    v = [rng.random() for _ in keys]
    s = sum(v) if normalized else 1.0
    return {k: x / s for k, x in zip(keys, v)}


# --------------------------------------------------------------- Section 2 / Proposition 8.3 example
def test_prop_8_3_example():
    A, B = Opinion.from_evidence(10, 0), Opinion.from_evidence(0, 10)
    assert (round(A.b, 3), round(A.u, 3)) == (0.833, 0.167)
    assert round(sl.degree_of_conflict(A, B), 3) == 0.579
    fused = sl.cumulative_fusion(A, B)
    coin = Opinion.from_evidence(10, 10)
    assert (round(fused.b, 3), round(fused.d, 3), round(fused.u, 3)) == (0.455, 0.455, 0.091)
    assert abs(fused.b - coin.b) < TOL and abs(fused.u - coin.u) < TOL


def test_def_8_4_separates_disagreement_from_coin():
    rA, rB = Reports(t=10), Reports(f=10)
    x = fused_contradiction([rA, rB])
    coin = fused_contradiction([Reports(t=10, f=10)])
    assert round(x.T, 3) == 0.455 and round(x.F, 3) == 0.455 and round(x.G, 3) == 0.091
    assert round(x.C, 3) == 0.579 and coin.C == 0


# --------------------------------------------------------------- Theorem 8.2
def test_theorem_8_2_sl_special_case():
    r = Reports(t=7, f=3)
    x, o = rnel_tuple(r), sl_opinion(r)
    assert abs(x.T - o.b) < TOL and abs(x.F - o.d) < TOL and abs(x.G - o.u) < TOL
    r2 = Reports(t=2, f=5)
    assert abs(rnel_tuple(r + r2).T - sl.cumulative_fusion(sl_opinion(r), sl_opinion(r2)).b) < TOL


def test_def_8_1_normalised():
    x = rnel_tuple(Reports(t=3, f=1, c=2, v=1, n=1, extra=(0.5,)))
    assert abs(x.total - 1) < TOL


# --------------------------------------------------------------- Theorem 5.2: (T, I, N, F)
@pytest.mark.parametrize("seed", range(200))
def test_theorem_5_2(seed):
    rng = random.Random(seed)
    x, y = rand_tuple("TINF", rng=rng), rand_tuple("TINF", rng=rng)
    assert op.tinf_not(op.tinf_not(x)) == x
    a = op.tinf_not(op.tinf_and(x, y))
    b = op.tinf_or(op.tinf_not(x), op.tinf_not(y))
    assert all(abs(a[k] - b[k]) < TOL for k in a)
    assert abs(op.mass(op.tinf_and(x, y)) - op.mass(x) * op.mass(y)) < TOL
    # on normalised inputs, pi(x and y) is SL multiplication with a = 0, pi(x or y) comultiplication with a = 1
    bx, ux, dx = op.coarsen(x)
    by, uy, dy = op.coarsen(y)
    m = sl.multiply(Opinion(bx, dx, ux, 0.0), Opinion(by, dy, uy, 0.0))
    t, u, f = op.coarsen(op.tinf_and(x, y))
    assert abs(t - m.b) < TOL and abs(f - m.d) < TOL and abs(u - m.u) < TOL
    c = sl.comultiply(Opinion(bx, dx, ux, 1.0), Opinion(by, dy, uy, 1.0))
    t, u, f = op.coarsen(op.tinf_or(x, y))
    assert abs(t - c.b) < 1e-9 and abs(f - c.d) < 1e-9


# --------------------------------------------------------------- Theorem 6.2: multi-uncertainty, n = 3
@pytest.mark.parametrize("seed", range(100))
def test_theorem_6_2(seed):
    rng = random.Random(seed)
    keys = ["T", "I", "N", "U1", "U2", "U3", "F"]
    x, y = rand_tuple(keys, rng=rng), rand_tuple(keys, rng=rng)
    a = op.mu_not(op.mu_and(x, y))
    b = op.mu_or(op.mu_not(x), op.mu_not(y))
    assert all(abs(a[k] - b[k]) < TOL for k in a)
    assert abs(op.mass(op.mu_and(x, y)) - 1) < TOL


# --------------------------------------------------------------- Section 8.3: RNEL operators
@pytest.mark.parametrize("seed", range(100))
def test_rnel_operators(seed):
    rng = random.Random(seed)
    keys = ["T", "C", "U", "N", "G", "F"]
    x, y = rand_tuple(keys, rng=rng), rand_tuple(keys, rng=rng)
    a = op.rnel_not(op.rnel_and(x, y))
    b = op.rnel_or(op.rnel_not(x), op.rnel_not(y))
    assert all(abs(a[k] - b[k]) < TOL for k in a)
    bx, ux, dx = op.coarsen(x)
    by, uy, dy = op.coarsen(y)
    m = sl.multiply(Opinion(bx, dx, ux, 0.0), Opinion(by, dy, uy, 0.0))
    t, u, f = op.coarsen(op.rnel_and(x, y))
    assert abs(t - m.b) < TOL and abs(f - m.d) < TOL


# --------------------------------------------------------------- ranking
def test_mu_key_reduces_to_tinf():
    x = {"T": 0.5, "I": 0.1, "N": 0.2, "F": 0.2}
    assert mu_key(x) == pytest.approx(tinf_key(x))
