"""Tests of rnel.copula against Section 8.3 of the book (script nnorm_choice.py) and Theorem 7."""
from fractions import Fraction as Fr

import numpy as np
import pytest

from rnel import copula as cop


def simplex_grid(n):
    return [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]


def random_copulas():
    return [cop.pi_copula, cop.upper_frechet, cop.lower_frechet, cop.frank(-5), cop.frank(3), cop.clayton(2),
            cop.gumbel(2), cop.fgm(0.7)]


def test_copula_boundary_conditions_and_frechet_bounds():
    g = np.linspace(0, 1, 11)
    for C in random_copulas():
        for u in g:
            assert C(u, 0) == pytest.approx(0, abs=1e-12) and C(0, u) == pytest.approx(0, abs=1e-12)
            assert C(u, 1) == pytest.approx(u, abs=1e-9) and C(1, u) == pytest.approx(u, abs=1e-9)
            for v in g:
                assert cop.lower_frechet(u, v) - 1e-12 <= C(u, v) <= cop.upper_frechet(u, v) + 1e-12


def test_copula_lookup_by_name():
    assert cop.copula("Pi") is cop.pi_copula and cop.copula("W") is cop.lower_frechet
    assert cop.copula("Frank", 5)(0.3, 0.4) == pytest.approx(cop.frank(5)(0.3, 0.4))
    assert cop.copula("survival:Clayton", 2)(0.3, 0.4) == pytest.approx(cop.survival(cop.clayton(2))(0.3, 0.4))
    with pytest.raises(ValueError):
        cop.copula("Frank")


@pytest.mark.theorem("Proposition 8.3.1")
def test_proposition_8_3_1_frechet_bracket_for_n_fold_compositions():
    rng = np.random.default_rng(0)
    fam = random_copulas()
    for _ in range(300):
        n = int(rng.integers(2, 6))
        xs = [tuple(rng.dirichlet([1, 1, 1])) for _ in range(n)]
        Cs = [fam[int(rng.integers(len(fam)))] for _ in range(n - 1)]
        T, I, F = cop.nnorm_n(xs, Cs)
        NW, NM = cop.frechet_bracket(xs)
        assert NW[0] - 1e-12 <= T <= NM[0] + 1e-12
        assert NM[1] - 1e-12 <= I <= NW[1] + 1e-12
        assert NM[2] - 1e-12 <= F <= NW[2] + 1e-12
        s = cop.classic_score((T, I, F))
        assert cop.classic_score(NW) - 1e-12 <= s <= cop.classic_score(NM) + 1e-12


@pytest.mark.theorem("Proposition 8.3.1")
def test_bracket_is_not_the_theorem9b_triple_example():
    x, y = (0.95, 0.05, 0.00), (0.45, 0.25, 0.30)  # criteria 1 and 3 of alternative A2 (Section 8.3.1)
    NW = cop.nnorm(x, y, cop.lower_frechet)
    NM = cop.nnorm(x, y, cop.upper_frechet)
    assert np.allclose(NW, (0.40, 0.30, 0.30)) and np.allclose(NM, (0.45, 0.25, 0.30))


@pytest.mark.theorem("Proposition 8.3.2")
def test_proposition_8_3_2_face_identity_and_non_symmetric_copulas_understate():
    g = [k / 20 for k in range(21)]
    for C in (cop.pi_copula, cop.upper_frechet, cop.lower_frechet, cop.frank(5.0), cop.clayton(2.0), cop.gumbel(2.0)):
        Ch = cop.survival(C)
        for a in g:
            for b in g:
                s = cop.surplus(C, (1 - a, a, 0.0), (1 - b, b, 0.0))
                assert s == pytest.approx(Ch(a, b) - C(a, b), abs=1e-12)
    # explicit case of the text: Clayton(2), x1 = x2 = (0.8, 0.2, 0)
    C = cop.clayton(2.0)
    assert C(0.2, 0.2) == pytest.approx(1 / 7)
    assert round(C(0.8, 0.8), 4) == 0.6860
    assert cop.surplus(C, (0.8, 0.2, 0), (0.8, 0.2, 0)) < 0
    assert not cop.is_radially_symmetric(C) and not cop.is_radially_symmetric(cop.gumbel(2.0))
    for C in (cop.pi_copula, cop.upper_frechet, cop.lower_frechet, cop.frank(-3), cop.fgm(0.5)):
        assert cop.is_radially_symmetric(C, tol=1e-9)


@pytest.mark.theorem("Proposition 8.3.2")
def test_grid_minima_of_the_surplus_reported_in_section_8_3_3():
    P = simplex_grid(20)
    m_clayton = min(cop.surplus(cop.clayton(2.0), a, b) for a in P for b in P)
    m_gumbel = min(cop.surplus(cop.gumbel(2.0), a, b) for a in P for b in P)
    assert round(m_clayton, 4) == -0.0569 and round(m_gumbel, 4) == -0.0267


@pytest.mark.theorem("Proposition 8.3.3")
def test_proposition_8_3_3_shuffle_is_symmetric_copula_with_negative_surplus():
    S = cop.shuffle_S()
    N = 24
    gq = [Fr(i, N) for i in range(N + 1)]
    for u in gq:
        assert S(u, Fr(0)) == 0 and S(Fr(0), u) == 0 and S(u, Fr(1)) == u and S(Fr(1), u) == u
    for i in range(N):
        for j in range(N):
            assert S(gq[i + 1], gq[j + 1]) - S(gq[i], gq[j + 1]) - S(gq[i + 1], gq[j]) + S(gq[i], gq[j]) >= 0
    for u in gq:
        for v in gq:
            assert S(u, v) == u + v - 1 + S(1 - u, 1 - v)
            assert S(u, v) == S(v, u)
    x = (Fr(1, 2), Fr(1, 4), Fr(1, 4))
    sigma = x[1] + x[1] - S(x[1], x[1]) - S(1 - x[2], 1 - x[2]) + S(x[0], x[0])
    assert sigma == Fr(-1, 4)


@pytest.mark.theorem("Proposition 8.3.4")
def test_proposition_8_3_4_lukasiewicz_is_conservative_and_closed_form():
    P = simplex_grid(20)
    worst_formula, minimum = 0.0, 1.0
    for a in P:
        for b in P:
            s = cop.surplus(cop.lower_frechet, a, b)
            worst_formula = max(worst_formula, abs(s - cop.surplus_W_closed_form(a, b)))
            minimum = min(minimum, s)
    assert worst_formula < 1e-12
    assert minimum > -1e-12


@pytest.mark.theorem("Theorem 7(a)", "Theorem 7(b)")
def test_theorem7_classical_frame_surplus_of_algebraic_and_minmax_nnorms():
    rng = np.random.default_rng(1)
    for _ in range(3000):
        x, y = rng.dirichlet([1, 1, 1]), rng.dirichlet([1, 1, 1])
        (T1, I1, F1), (T2, I2, F2) = x, y
        assert cop.surplus(cop.pi_copula, x, y) == pytest.approx(I1 * F2 + I2 * F1, abs=1e-12)
        u1, u2 = 1 - F1, 1 - F2
        sM = cop.surplus(cop.upper_frechet, x, y)
        assert sM == pytest.approx(max(I1, I2) - (min(u1, u2) - min(T1, T2)), abs=1e-12)
        assert sM >= -1e-12


@pytest.mark.theorem("Table 8.3.3")
def test_table_8_3_3_frank_sweep_scores_and_ranking_reversal():
    alts = {
        "A1": [(0.70, 0.20, 0.10), (0.70, 0.15, 0.15), (0.70, 0.20, 0.10)],
        "A2": [(0.95, 0.05, 0.00), (0.95, 0.00, 0.05), (0.45, 0.25, 0.30)],
        "A3": [(0.80, 0.10, 0.10), (0.85, 0.10, 0.05), (0.60, 0.25, 0.15)],
        "A4": [(0.90, 0.05, 0.05), (0.60, 0.30, 0.10), (0.72, 0.13, 0.15)],
    }
    table = {
        "W": (cop.lower_frechet, [0.4000, 0.5667, 0.5000, 0.4800]),
        "-20": (cop.frank(-20), [0.4021, 0.5667, 0.5001, 0.4802]),
        "-5": (cop.frank(-5), [0.4338, 0.5716, 0.5159, 0.4984]),
        "-1": (cop.frank(-1), [0.4994, 0.5879, 0.5618, 0.5457]),
        "Pi": (cop.pi_copula, [0.5252, 0.5945, 0.5807, 0.5647]),
        "1": (cop.frank(1), [0.5524, 0.6015, 0.6009, 0.5848]),
        "5": (cop.frank(5), [0.6436, 0.6221, 0.6677, 0.6510]),
        "20": (cop.frank(20), [0.7427, 0.6331, 0.7259, 0.7090]),
        "M": (cop.upper_frechet, [0.7833, 0.6333, 0.7333, 0.7167]),
    }
    for C, expected in table.values():
        got = [round(cop.classic_score(cop.nnorm_n(v, C)), 4) for v in alts.values()]
        assert got == expected
    # Frank N-norms are order-free (Frank's theorem): all orderings of the criteria give the same triple
    import itertools
    for v in alts.values():
        outs = [cop.nnorm_n(list(p), cop.frank(5)) for p in itertools.permutations(v)]
        assert max(abs(a - b) for o in outs for a, b in zip(o, outs[0])) < 1e-12


def test_tail_dependence_and_kendall_tau():
    assert round(cop.tail_dependence(cop.clayton(2.0), "lower"), 4) == 0.7071
    assert round(cop.tail_dependence(cop.gumbel(2.0), "upper"), 4) == 0.5858
    assert round(cop.tail_dependence(cop.frank(5.0), "lower"), 4) == 0.0
    assert cop.kendall_tau("clayton", 2) == pytest.approx(0.5)
    th = cop.frank_theta_from_tau(0.5)
    assert cop.kendall_tau("frank", th) == pytest.approx(0.5, abs=1e-6)
    assert round(th, 3) == 5.736
