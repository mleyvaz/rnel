import numpy as np
import pytest

from rnel.existential import (
    UnfaithfulTripleWarning,
    belnap_masses,
    dempster_conflict,
    existential_triple,
    glut,
    glut_test,
    in_faithful_region,
    isotonic_fit,
    isotonic_predict,
    mondrian_calibrate,
    venn_abers,
)


def random_profiles(rng, n_claims, kmin=1, kmax=8):
    ks = rng.integers(kmin, kmax + 1, size=n_claims)
    owner = np.repeat(np.arange(n_claims), ks)
    P = rng.dirichlet([0.7, 0.7, 0.7], size=owner.size)
    return P[:, 0], P[:, 1], P[:, 2], owner, ks


def test_glut_equals_dempster_conflict():
    rng = np.random.default_rng(0)
    s, r, n, owner, _ = random_profiles(rng, 2000)
    T, I, F = existential_triple(s, r, n, owner)
    assert np.max(np.abs(glut(T, I, F) - dempster_conflict(s, r, n, owner))) < 1e-12


def test_extreme_probabilities_and_empty_claims():
    s = np.array([1.0, 0.0, 0.0, 0.5])
    r = np.array([0.0, 1.0, 0.0, 0.5])
    n = np.array([0.0, 0.0, 1.0, 0.0])
    owner = np.array([0, 0, 1, 3])
    T, I, F = existential_triple(s, r, n, owner, n_claims=5)
    assert np.allclose(T, [1, 0, 0, 0.5, 0]) and np.allclose(F, [1, 0, 0, 0.5, 0]) and np.allclose(I, [0, 1, 1, 0, 1])
    assert np.allclose(glut(T, I, F), [1, 0, 0, 0, 0])
    assert np.allclose(dempster_conflict(s, r, n, owner, n_claims=5), glut(T, I, F))


def test_glut_matches_monte_carlo_frequency():
    rng = np.random.default_rng(1)
    n_claims, R = 40, 20000
    s, r, n, owner, ks = random_profiles(rng, n_claims, 1, 6)
    T, I, F = existential_triple(s, r, n, owner)
    g = glut(T, I, F)
    cum = np.cumsum(np.stack([s, r, n], 1), axis=1)
    u = rng.random((R, owner.size))
    stance = (u[..., None] > cum[None]).sum(-1)  # 0 support, 1 refute, 2 insufficient
    sup = np.zeros((R, n_claims), bool)
    ref = np.zeros((R, n_claims), bool)
    for c in range(n_claims):
        cols = owner == c
        sup[:, c] = (stance[:, cols] == 0).any(1)
        ref[:, c] = (stance[:, cols] == 1).any(1)
    freq = (sup & ref).mean(0)
    se = np.sqrt(np.maximum(g * (1 - g), 1e-12) / R)
    assert np.all(np.abs(freq - g) <= 4.5 * se + 1e-9)
    assert np.allclose(sup.mean(0), T, atol=0.02) and np.allclose(ref.mean(0), F, atol=0.02)


def test_single_source_has_no_glut():
    rng = np.random.default_rng(2)
    P = rng.dirichlet([1, 1, 1], size=500)
    T, I, F = existential_triple(P[:, 0], P[:, 1], P[:, 2], np.arange(500))
    assert np.allclose(T + I + F, 1.0, atol=1e-12)


def test_belnap_masses_nonnegative_and_sum_to_one():
    rng = np.random.default_rng(3)
    s, r, n, owner, _ = random_profiles(rng, 3000)
    T, I, F = existential_triple(s, r, n, owner)
    t, f, b, nn = belnap_masses(T, I, F)
    M = np.stack([t, f, b, nn])
    assert np.all(M >= 0)
    assert np.allclose(M.sum(0), 1.0, atol=1e-12)
    assert np.all(in_faithful_region(T, I, F))


def test_belnap_masses_flags_unfaithful_triples():
    T, I, F = np.array([0.1]), np.array([0.5]), np.array([0.9])  # g = 0.5 > T -> t < 0
    with pytest.raises(ValueError):
        belnap_masses(T, I, F)
    with pytest.warns(UnfaithfulTripleWarning):
        t, f, b, nn = belnap_masses(T, I, F, strict=False)
    assert t[0] < 0


def test_invalid_rows_rejected():
    with pytest.raises(ValueError):
        existential_triple([0.5], [0.6], [0.1], [0])


def test_isotonic_fit_basic():
    m = isotonic_fit([1, 2, 3, 4, 5], [0, 1, 0, 1, 1])
    assert np.allclose(m.y, [0, 0.5, 0.5, 1, 1])
    assert np.all(np.diff(isotonic_predict(m, np.linspace(0, 6, 50))) >= -1e-15)
    m2 = isotonic_fit([1, 1, 2], [1, 0, 0])  # ties pooled
    assert np.allclose(m2.y, [1 / 3, 1 / 3])


def test_venn_abers_matches_brute_force():
    rng = np.random.default_rng(4)
    n = 300
    cs = np.round(rng.normal(size=n), 1)  # rounding creates ties
    cl = (rng.random(n) < 1 / (1 + np.exp(-2 * cs))).astype(float)
    ts = np.concatenate([np.round(rng.normal(size=150), 1), rng.normal(size=150), [cs.min() - 1, cs.max() + 1]])
    p0, p1, p = venn_abers(cs, cl, ts)
    b0 = np.array([isotonic_predict(isotonic_fit(np.append(cs, x), np.append(cl, 0.0)), x) for x in ts])
    b1 = np.array([isotonic_predict(isotonic_fit(np.append(cs, x), np.append(cl, 1.0)), x) for x in ts])
    assert np.max(np.abs(p0 - b0)) < 1e-12 and np.max(np.abs(p1 - b1)) < 1e-12
    assert np.all(p0 <= p1 + 1e-15)
    assert np.allclose(p, p1 / (1 - p0 + p1))
    assert np.all((p >= p0 - 1e-12) & (p <= p1 + 1e-12))


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_glut_test_validity(alpha):
    rng = np.random.default_rng(5)
    reps, n = 5000, 50
    cal = rng.beta(0.5, 3.0, size=(reps, n))
    test = rng.beta(0.5, 3.0, size=reps)
    pv = np.array([glut_test(cal[i], test[i]) for i in range(reps)])
    rate = np.mean(pv <= alpha)
    se = np.sqrt(alpha * (1 - alpha) / reps)
    assert rate <= alpha + 3 * se
    assert np.all((pv > 0) & (pv <= 1))


def test_glut_test_values():
    assert np.allclose(glut_test([0.1, 0.2, 0.3], [0.0, 0.25, 0.9, 0.2]), [1.0, 0.5, 0.25, 0.75])


def _ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def _dependent_claims(rng, n_claims):
    """Claims checked by 1-5 sources that share one latent signal (copying), so noisy-OR T is overconfident."""
    k = rng.integers(1, 6, size=n_claims)
    y = (rng.random(n_claims) < 0.4).astype(float)
    z = (2 * y - 1) * 0.8 + rng.normal(size=n_claims)  # shared evidence of the claim
    owner = np.repeat(np.arange(n_claims), k)
    zk = z[owner] + 0.3 * rng.normal(size=owner.size)
    s = 0.8 / (1 + np.exp(-2.5 * zk))
    r = 0.8 - s
    nn = np.full_like(s, 0.2)
    T, I, F = existential_triple(s, r, nn, owner, n_claims)
    return T, y, k


def test_mondrian_calibrate_reduces_error_per_stratum():
    rng = np.random.default_rng(6)
    Tc, yc, kc = _dependent_claims(rng, 15000)
    Tt, yt, kt = _dependent_claims(rng, 15000)
    p, info = mondrian_calibrate(Tc, yc, kc, Tt, kt, return_info=True)
    assert np.all((p >= 0) & (p <= 1))
    assert set(info.values()) == {"isotonic"}
    for h in range(1, 6):
        m = kt == h
        raw, cal = _ece(Tt[m], yt[m]), _ece(p[m], yt[m])
        assert cal < raw, (h, raw, cal)
        assert cal < 0.05


def test_mondrian_fallbacks():
    g_cal = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.5, 0.6])
    y_cal = np.array([0, 1, 0, 1, 0, 1, 1.0])
    st_cal = np.array([1, 1, 1, 1, 1, 2, 2])
    p, info = mondrian_calibrate(g_cal, y_cal, st_cal, [0.5, 0.5, 0.95], [1, 2, 7], min_size=5, return_info=True)
    assert info == {1: "isotonic", 2: "base_rate", 7: "global"}
    assert p[1] == 1.0  # single-class stratum -> base rate
    assert np.all((p >= 0) & (p <= 1))
