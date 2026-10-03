import random

import pytest

from rnel.neutro_stats import (NNumber, credal_interval, decompose, estimate, estimate_sources,
                               interval_sum)
from rnel.tuple import Reports

R = Reports(t=6, f=2, c=3, v=1, n=2)       # S = 6 + 2 + 3 + 1 + 2 + 2 = 16


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def test_estimate_form():
    e = estimate(R)
    assert e.S == 16
    assert close(e.x.a, 6 / 16)
    assert {k: round(v * 16, 9) for k, v in e.x.terms.items()} == {"C": 3, "U": 1, "N": 2, "G": 2}
    assert all(close(e.not_x.terms[k], -v) for k, v in e.x.terms.items())


@pytest.mark.theorem("Proposition 12.7.1")
def test_credal_reading_is_idm_interval():
    e = estimate(R)
    lo, hi = credal_interval(R)
    assert close(e.x.lo, lo) and close(e.x.hi, hi)
    assert close(lo, 6 / 16) and close(hi, 14 / 16)


def test_matches_neutro_credal_idm_without_indeterminate_reports():
    np = pytest.importorskip("numpy")
    from rnel.neutro_credal import idm_triple
    r = Reports(t=7, f=3)
    T, I, F = idm_triple(np.array([7.0, 3.0]), s=2.0)[0]
    e = estimate(r, W=2.0)
    assert close(e.x.lo, T) and close(e.x.hi, 1 - F) and close(e.x.width, I)


@pytest.mark.theorem("Proposition 12.7.1")
def test_coherence_exact_versus_interval_arithmetic():
    e = estimate(R)
    tot = e.total
    assert tot.terms == {} and close(tot.a, 1.0)                      # symbols cancel: exactly 1
    lo, hi = interval_sum(e.x, e.not_x)                               # symbols forgotten
    assert close(lo, 1 - e.x.width) and close(hi, 1 + e.x.width)      # [1 - w, 1 + w]


@pytest.mark.theorem("Section 12.7.4 (glut and gap readings)")
def test_glut_and_gap_readings():
    g = estimate(R, glut=True).total
    assert g.terms == {} and close(g.a, 1 + 3 / 16)
    p = estimate(R, gap=True).total
    assert p.terms == {} and close(p.a, 1 - 2 / 16)
    b = estimate(R, glut=True, gap=True)
    assert close(b.total.a, 1 + 3 / 16 - 2 / 16) and close(b.glut, 3 / 16) and close(b.gap, 2 / 16)
    # under the glut reading the contradictory mass is no longer indeterminate: it lifts both lower bounds
    assert close(b.x.lo, (6 + 3) / 16) and close(b.not_x.lo, (2 + 3) / 16)


@pytest.mark.theorem("Section 12.7.4 (glut and gap readings)")
def test_glut_has_no_probability_on_the_same_frame():
    e = estimate(R, glut=True)
    # every completion of the symbols gives p(x) + p(not x) > 1: no single probability on {x, not x}
    for _ in range(200):
        vals = {k: random.random() for k in e.x.terms}
        assert e.x.at(vals) + e.not_x.at(vals) > 1


@pytest.mark.theorem("Proposition 12.7.1")
def test_decomposition_by_type_and_source():
    e = estimate_sources({"A": Reports(t=4, f=0, c=2), "B": Reports(t=0, f=3, c=1, n=2)})
    S = 4 + 3 + 2 + 1 + 2 + 2
    assert close(e.S, S)
    by_type = decompose(e.x, by="type")
    assert close(by_type["C"], 3 / S) and close(by_type["N"], 2 / S) and close(by_type["G"], 2 / S)
    by_src = decompose(e.x, by="source")
    assert close(by_src["A"], 2 / S) and close(by_src["B"], 3 / S) and close(by_src["G"], 2 / S)
    rel = decompose(e.x, by="type", relative=True)
    assert close(sum(rel.values()), 1.0)
    # pooling the sources gives the same range as keeping them apart
    pooled = estimate(Reports(t=4, f=3, c=3, n=2))
    assert close(pooled.x.lo, e.x.lo) and close(pooled.x.hi, e.x.hi)


@pytest.mark.theorem("Section 12.7.1 (affine product, coefficient 2 r_x r_y)")
def test_product_is_enclosing_and_tighter_than_old_msnn():
    random.seed(1)
    x = NNumber(0.2, {"a": 0.3, "b": 0.1})
    y = NNumber(0.5, {"a": 0.2, "c": 0.4})
    z = x * y
    for _ in range(2000):
        v = {k: random.random() for k in "abc"}
        assert z.lo - 1e-12 <= x.at(v) * y.at(v) <= z.hi + 1e-12
    rx, ry = x.width / 2, y.width / 2
    old_width = z.width + 2 * rx * ry          # the affine paper's MSNN used 4 rx ry instead of 2 rx ry
    assert z.width < old_width


@pytest.mark.theorem("Section 12.7.1 (affine bijection I = (1+eps)/2)")
def test_affine_operations_and_scalar_division():
    x = NNumber(1.0, {"a": 2.0})
    assert (x - x).terms == {} and close((x - x).a, 0.0)
    assert close((x / 2).hi, 1.5)
    assert close((3 - x).lo, 0.0)
    with pytest.raises(TypeError):
        x / x


# ----------------------------------------------------------------------------- affine bijection (Section 12.7.1)
@pytest.mark.theorem("Section 12.7.1 (affine bijection I = (1+eps)/2)")
def test_affine_bijection_round_trip_and_same_range():
    import random as _r
    from rnel.neutro_stats import AffineForm, I_from_eps, NNumber, eps_from_I, from_affine, to_affine
    g = _r.Random(7)
    for _ in range(500):
        x = NNumber(g.uniform(-1, 1), {k: g.uniform(-1, 1) for k in "CUNG"[: g.randint(0, 4)]})
        f = to_affine(x)
        assert f.range() == pytest.approx(x.range())
        y = from_affine(f)
        assert y.a == pytest.approx(x.a) and set(y.terms) == set(x.terms)
        assert all(y.terms[k] == pytest.approx(v) for k, v in x.terms.items())
        I = {k: g.random() for k in x.terms}
        eps = {k: eps_from_I(v) for k, v in I.items()}
        assert x.at(I) == pytest.approx(f.x0 + sum(c * eps[k] for k, c in f.coeffs.items()))
        assert all(I_from_eps(eps[k]) == pytest.approx(I[k]) for k in I)


@pytest.mark.theorem("Section 12.7.1 (affine product, coefficient 2 r_x r_y)")
def test_product_parity_with_reference_affine_arithmetic():
    import random as _r
    from rnel.neutro_stats import NNumber, to_affine
    g = _r.Random(11)
    for _ in range(300):
        x = NNumber(g.uniform(-1, 1), {k: g.uniform(-1, 1) for k in "CUN"})
        y = NNumber(g.uniform(-1, 1), {k: g.uniform(-1, 1) for k in "UNG"})
        p_nn = to_affine(x * y)
        p_af = to_affine(x) * to_affine(y)
        assert p_nn.x0 == pytest.approx(p_af.x0)
        shared = {k for k in p_nn.coeffs if not (isinstance(k, tuple) and k[0] == "err")}
        assert all(p_nn.coeffs[k] == pytest.approx(p_af.coeffs.get(k, 0.0)) for k in shared)
        assert p_nn.radius == pytest.approx(p_af.radius)
        assert p_nn.range() == pytest.approx(p_af.range())
