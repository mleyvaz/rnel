"""Tests of rnel.credal (experimental)."""
import itertools
import random

import numpy as np
import pytest

pytest.importorskip("sklearn")

from rnel import conflict as cf
from rnel import credal as cr

TOL = 1e-9


def rand_members(rng, M=7, n=11, K=4):
    P = rng.gamma(0.7, size=(M, n, K))
    return P / P.sum(axis=2, keepdims=True)


# ----------------------------------------------------------------------------- representation only
@pytest.mark.theorem("Theorem 1")
def test_neutrosophic_identities():
    rng = np.random.default_rng(0)
    for method in ("hull", "trimmed", "quantile"):
        l, u = cr.envelope(rand_members(rng), method, alpha=0.3, q=0.1)
        T, I, F = cr.to_neutrosophic(l, u)
        assert np.allclose(T + I + F, 1)
        assert np.allclose(T, l) and np.allclose(I, u - l) and np.allclose(F, 1 - u)
        assert (T >= -TOL).all() and (I >= -TOL).all() and (F >= -TOL).all()


@pytest.mark.theorem("Theorem 1")
def test_bijection_with_normalised_pair():
    rng = np.random.default_rng(1)
    # (l, u) -> (T, I, F) -> (l, u)
    a, b = rng.random((2, 1000))
    l, u = np.minimum(a, b), np.maximum(a, b)
    l2, u2 = cr.from_neutrosophic(*cr.to_neutrosophic(l, u))
    assert np.allclose(l, l2) and np.allclose(u, u2)
    # normalised (T, I, F) -> (l, u) -> (T, I, F)
    X = rng.dirichlet([1, 1, 1], size=1000)
    T, I, F = X.T
    T2, I2, F2 = cr.to_neutrosophic(*cr.from_neutrosophic(T, I, F))
    assert np.allclose(T, T2) and np.allclose(I, I2) and np.allclose(F, F2)
    # images are exactly the valid sets
    with pytest.raises(ValueError):
        cr.to_neutrosophic(np.array([0.6]), np.array([0.4]))
    with pytest.raises(ValueError):
        cr.from_neutrosophic(np.array([0.5]), np.array([0.5]), np.array([0.5]))


def test_hull_envelope_is_min_max_and_coherent():
    rng = np.random.default_rng(2)
    P = rand_members(rng)
    l, u = cr.envelope(P, "hull")
    assert np.allclose(l, P.min(0)) and np.allclose(u, P.max(0))
    lc, uc = cr.coherent_bounds(l, u)
    assert np.allclose(lc, l) and np.allclose(uc, u)  # envelope of genuine distributions is reachable
    assert (l.sum(1) <= 1 + TOL).all() and (u.sum(1) >= 1 - TOL).all()


def test_trimmed_nested_in_hull():
    rng = np.random.default_rng(3)
    P = rand_members(rng, M=20)
    l, u = cr.envelope(P, "hull")
    lt, ut = cr.envelope(P, "trimmed", alpha=0.25)
    assert (lt >= l - TOL).all() and (ut <= u + TOL).all()
    l0, u0 = cr.envelope(P, "trimmed", alpha=0.0)
    assert np.allclose(l0, l) and np.allclose(u0, u)


# ----------------------------------------------------------------------------- the ensemble
@pytest.fixture(scope="module")
def fitted():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=400, n_features=8, n_informative=5, n_classes=3,
                               random_state=0)
    y = np.array(["a", "b", "c"])[y]
    ce = cr.CredalEnsemble(random_state=0).fit(X[:300], y[:300])
    return ce, X[300:], y[300:]


def test_credal_ensemble_bounds(fitted):
    ce, X, _ = fitted
    P = ce.member_proba(X)
    assert P.shape == (100, len(X), 3)
    assert np.allclose(P.sum(axis=2), 1)
    l, u = ce.bounds(X)
    assert np.allclose(l, P.min(0)) and np.allclose(u, P.max(0))
    p = ce.predict_proba(X)
    assert np.allclose(p, ce.estimator_.predict_proba(X))  # the forest's own prediction
    assert ((l <= p + TOL) & (p <= u + TOL)).all()
    assert np.allclose(ce.width(X), ce.neutrosophic(X)[1])


def test_evidence_modes(fitted):
    ce, X, _ = fitted
    ev = ce.evidence(X, "votes")
    assert np.allclose(ev.sum(1), 100)
    lf = ce.evidence(X, "leaf")
    assert (lf >= 0).all() and (lf.sum(1) >= 5 - 1e-9).all()  # min_samples_leaf = 5 (bootstrap weights)
    assert np.allclose(ce.evidence(X, "mean", scale=3.0).sum(1), 3.0)


# ----------------------------------------------------------------------------- conflict between views
def test_sup_gap_matches_rnel_conflict():
    rng = random.Random(4)
    for _ in range(300):
        V, K = rng.randint(1, 4), rng.randint(2, 5)
        E = np.array([[rng.expovariate(0.3) for _ in range(K)] for _ in range(V)])
        assert abs(cr.sup_gap_views(E[:, None, :])[0] - cf.sup_gap(E.tolist())) < 1e-9


def test_binary_c_star_matches_conflict_state_and_multiclass_reduces():
    rng = np.random.default_rng(5)
    for _ in range(300):
        V = rng.integers(1, 5)
        E = rng.exponential(3, size=(V, 1, 2))
        ref = cf.c_star([tuple(e[0]) for e in E])
        assert abs(cr.c_star_one_vs_rest(E, 0)[0] - ref) < 1e-9
        assert abs(cr.c_star_multiclass(E)[0] - ref) < 1e-9  # K = 2: sup-norm C* = binary C*


@pytest.mark.theorem("Section 12.4 (C* order-free)")
def test_c_star_between_views_is_order_free():
    rng = np.random.default_rng(6)
    E = rng.exponential(2, size=(4, 50, 3))
    ref_m = cr.c_star_multiclass(E)
    ref_s = cr.evidential_state(E, 1)
    for perm in itertools.permutations(range(4)):
        assert np.allclose(cr.c_star_multiclass(E[list(perm)]), ref_m)
        st = cr.evidential_state(E[list(perm)], 1)
        for k in ("a", "b", "u", "c"):
            assert np.allclose(st[k], ref_s[k])


def test_evidential_state_identities():
    rng = np.random.default_rng(7)
    E = rng.exponential(2, size=(2, 200, 4))
    cls = rng.integers(0, 4, size=200)
    st = cr.evidential_state(E, cls)
    assert np.allclose(st["a"] + st["b"] + st["u"] + st["c"], 1)
    assert np.allclose(st["T"], st["a"] + st["c"]) and np.allclose(st["F"], st["b"] + st["c"])
    assert np.allclose(st["I"], st["u"] + st["c"])
    assert np.allclose(st["T"] + st["I"] + st["F"], 1 + 2 * st["c"])
    assert min(st[k].min() for k in "abuc") >= -TOL


def test_states_and_decisions_on_canonical_cases():
    W = 2.0
    agree = np.array([[[10.0, 0, 0]], [[10.0, 0, 0]]])     # both views support class 0
    clash = np.array([[[10.0, 0, 0]], [[0, 10.0, 0]]])     # views support different classes
    empty = np.array([[[0.1, 0, 0]], [[0.1, 0, 0]]])        # almost no evidence
    against = np.array([[[0, 10.0, 0]], [[0, 0, 10.0]]])   # both views against class 0
    s = {k: cr.evidential_state(v, 0, W) for k, v in
         dict(agree=agree, clash=clash, empty=empty, against=against).items()}
    assert s["agree"]["c"][0] == 0 and s["agree"]["a"][0] > 0.9
    assert s["clash"]["c"][0] > 0.8
    assert s["empty"]["u"][0] > 0.9
    assert s["against"]["b"][0] > 0.9 and s["against"]["c"][0] == 0  # rejection is not conflict
    assert cr.decide(s["agree"])[0] == "answer"
    assert cr.decide(s["clash"])[0] == "review_sources"
    assert cr.decide(s["empty"])[0] == "abstain_seek_data"
    assert cr.decide(s["against"])[0] == "reject"
    assert cr.c_star_multiclass(against)[0] > 0.8  # multiclass C* sees the clash between classes 1 and 2
