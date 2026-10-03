"""Tests of rnel.dsmt against published worked examples and properties.

Published sources (numbers copied from the papers; page/section given in each test):
  [SD]  Smarandache and Dezert, "Proportional conflict redistribution rules for information fusion", arXiv:cs/0408064v3
        (2005); also Chapter 1 of Advances and Applications of DSmT for Information Fusion, vol. 2 (2006).
  [MO]  Martin and Osswald, "A new generalization of the proportional conflict redistribution rule stable in terms of
        decision", arXiv:0806.1797; Chapter 2 of the same vol. 2 (2006).
  [DS08] Dezert and Smarandache, "A new probabilistic transformation of belief mass assignment", Fusion 2008,
        arXiv:0807.3669.
  [DDS21] Dezert, Dezert and Smarandache, "Improvement of proportional conflict redistribution rules of combination
        of basic belief assignments", JAIF 16(1), 2021 (Example 2, as reproduced and tested in the public repository
        github.com/mleyvaz/dsmt-rag-pcr6, test_pcr6_plus.py).
"""
import itertools
import math
import random

import pytest

from rnel import Reports, rnel_tuple
from rnel.dsmt import (
    RULES, Frame, TotalConflictError, bba_to_triple, bba_to_tuple, belief, belnap_probability, betp, binary_frame,
    conflict, conjunctive, dempster, dsmc, dsmp, dubois_prade, opinion_to_bba, pcr5, pcr5_shares, pcr6, pcr6_plus,
    pcr6_shares, plausibility, reports_to_bba, sequential, tbm, triple_to_bba, tuple_to_bba, yager,
)
from rnel.sl import Opinion


def close(fr, m, expected, places=6):
    got = {fr.element(k): v for k, v in expected.items()}
    keys = set(got) | {k for k, v in m.items() if v > 1e-12}
    for k in keys:
        assert m.get(k, 0.0) == pytest.approx(got.get(k, 0.0), abs=0.5 * 10 ** -places), (fr.label(k), m.get(k), got.get(k))


def is_bba(m):
    assert all(v >= -1e-12 for v in m.values())
    assert math.fsum(m.values()) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- frames
def test_hyper_power_set_sizes():
    # |D^Theta| (with the empty set) = 5, 19, 167 for n = 2, 3, 4 (Dedekind numbers minus one; DSmT vol. 1).
    assert [len(Frame("AB"[:2]).hyper_power_set()), len(Frame("ABC").hyper_power_set()),
            len(Frame("ABCD").hyper_power_set())] == [5, 19, 167]
    assert len(Frame("ABC", model="shafer").hyper_power_set()) == 8   # 2^3


def test_parser_and_labels():
    fr = Frame("ABC")
    assert fr.element("A&B|C") == (fr.atom("A") & fr.atom("B")) | fr.atom("C")
    assert fr.element("(A|B)&C") == (fr.atom("A") | fr.atom("B")) & fr.atom("C")
    assert fr.label(fr.element("(A|B)&C"), ascii=True) == "(A&C)|(B&C)"
    assert fr.element({"A", "B"}) == fr.atom("A") | fr.atom("B")
    assert fr.element("Theta") == fr.theta
    sh = Frame("ABC", model="shafer")
    assert not sh.element("A&B")
    h = Frame("ABC", model="hybrid", empty=[("A", "B")])
    assert not h.element("A&B") and h.element("A&C")


# --------------------------------------------------------------------------- published examples
def test_zadeh_all_rules_SD_12_3():
    """[SD] Section 12.3 (Zadeh's example): every rule as printed."""
    fr = Frame("ABC", model="shafer")
    m1, m2 = fr.bba({"A": 0.9, "C": 0.1}), fr.bba({"B": 0.9, "C": 0.1})
    assert conflict(fr, [m1, m2]) == pytest.approx(0.99)
    close(fr, dempster(fr, [m1, m2]), {"C": 1.0})
    close(fr, tbm(fr, [m1, m2]), {"empty": 0.99, "C": 0.01})
    close(fr, yager(fr, [m1, m2]), {"Theta": 0.99, "C": 0.01})
    close(fr, dubois_prade(fr, [m1, m2]), {"A|B": 0.81, "A|C": 0.09, "B|C": 0.09, "C": 0.01})
    close(fr, pcr5(fr, [m1, m2]), {"A": 0.486, "B": 0.486, "C": 0.028})
    close(fr, pcr6(fr, [m1, m2]), {"A": 0.486, "B": 0.486, "C": 0.028})
    free = Frame("ABC")
    f1, f2 = free.bba({"A": 0.9, "C": 0.1}), free.bba({"B": 0.9, "C": 0.1})
    close(free, dsmc(free, [f1, f2]), {"A&B": 0.81, "A&C": 0.09, "B&C": 0.09, "C": 0.01})


def test_conjunctive_SD_2_2_2():
    """[SD] Section 2.2.2: conjunctive consensus on Shafer's and on the free model."""
    for model, key in (("shafer", "empty"), ("free", "t1&t2")):
        fr = Frame(["t1", "t2"], model=model)
        m1 = fr.bba({"t1": 0.1, "t2": 0.2, "Theta": 0.7})
        m2 = fr.bba({"t1": 0.4, "t2": 0.3, "Theta": 0.3})
        close(fr, conjunctive(fr, [m1, m2]), {key: 0.11, "t1": 0.35, "t2": 0.33, "Theta": 0.21})


@pytest.mark.parametrize("m2, expected", [
    ({"A": 0.0, "B": 0.3, "Theta": 0.7}, {"A": 0.54, "B": 0.18, "Theta": 0.28}),   # [SD] 11.1.1
    ({"A": 0.2, "B": 0.3, "Theta": 0.5}, {"A": 0.62, "B": 0.18, "Theta": 0.20}),   # [SD] 11.1.2
])
def test_pcr5_two_sources_SD_11_1(m2, expected):
    fr = Frame("AB", model="shafer")
    m1 = fr.bba({"A": 0.6, "Theta": 0.4})
    close(fr, pcr5(fr, [m1, fr.bba(m2)]), expected)


def test_pcr5_and_dempster_SD_11_1_3():
    """[SD] 11.1.3: PCR5 0.584/0.366/0.05; Dempster 0.579/0.355/0.066 (printed to three decimals)."""
    fr = Frame("AB", model="shafer")
    m1, m2 = fr.bba({"A": 0.6, "B": 0.3, "Theta": 0.1}), fr.bba({"A": 0.2, "B": 0.3, "Theta": 0.5})
    close(fr, pcr5(fr, [m1, m2]), {"A": 0.584, "B": 0.366, "Theta": 0.05})
    close(fr, dempster(fr, [m1, m2]), {"A": 0.579, "B": 0.355, "Theta": 0.066}, places=3)
    # neutrality of the vacuous BBA ([SD] 11.5)
    close(fr, pcr5(fr, [m1, m2, fr.bba({"Theta": 1.0})]), {"A": 0.584, "B": 0.366, "Theta": 0.05})


def test_bayesian_three_classes_SD_12_1():
    fr = Frame("ABC", model="shafer")
    m1, m2 = fr.bba({"A": 0.6, "B": 0.3, "C": 0.1}), fr.bba({"A": 0.4, "B": 0.4, "C": 0.2})
    assert conflict(fr, [m1, m2]) == pytest.approx(0.62)
    close(fr, pcr5(fr, [m1, m2]), {"A": 0.574571, "B": 0.335429, "C": 0.090000})
    close(fr, dempster(fr, [m1, m2]), {"A": 0.631579, "B": 0.315789, "C": 0.052632})


def test_SD_12_2():
    fr = Frame("AB", model="shafer")
    m1, m2 = fr.bba({"A": 0.7, "B": 0.1, "Theta": 0.2}), fr.bba({"A": 0.5, "B": 0.4, "Theta": 0.1})
    assert conflict(fr, [m1, m2]) == pytest.approx(0.33)
    # printed 0.739849 / 0.240151; the exact values are 0.7398485 / 0.2401515 (last printed digit rounded up)
    close(fr, pcr5(fr, [m1, m2]), {"A": 0.739849, "B": 0.240151, "Theta": 0.02}, places=5)
    close(fr, dempster(fr, [m1, m2]), {"A": 0.776119, "B": 0.194030, "Theta": 0.029851})


def test_hybrid_model_SD_12_4():
    """[SD] 12.4: hybrid model with A∩B = ∅, A∩C and B∩C non-empty."""
    fr = Frame("ABC", model="hybrid", empty=[("A", "B")])
    m1, m2 = fr.bba({"A": 0.5, "B": 0.4, "C": 0.1}), fr.bba({"A": 0.6, "B": 0.2, "C": 0.2})
    close(fr, conjunctive(fr, [m1, m2]), {"A": 0.30, "B": 0.08, "C": 0.02, "empty": 0.34, "A&C": 0.16, "B&C": 0.10})
    close(fr, pcr5(fr, [m1, m2]), {"A": 0.51543, "B": 0.20457, "C": 0.02, "A&C": 0.16, "B&C": 0.10}, places=5)


def test_three_sources_SD_11_4():
    """[SD] 11.4: conjunctive consensus of three sources and two partial PCR5 redistributions printed there."""
    fr = Frame("AB", model="shafer")
    ms = [fr.bba({"A": 0.6, "B": 0.3, "Theta": 0.1}), fr.bba({"A": 0.2, "B": 0.3, "Theta": 0.5}),
          fr.bba({"A": 0.4, "B": 0.4, "Theta": 0.2})]
    close(fr, conjunctive(fr, ms), {"A": 0.284, "B": 0.182, "Theta": 0.010, "empty": 0.524})
    A, B, AB = fr.element("A"), fr.element("B"), fr.theta
    # m1(A) m2(A∪B) m3(B) = 0.120 goes to A and B in proportion 0.6 : 0.4 (A∪B is absorbed)
    sh = pcr5_shares((A, AB, B), (0.6, 0.5, 0.4))
    assert sh[A] == pytest.approx(0.072) and sh[B] == pytest.approx(0.048) and AB not in sh
    # m2(A) m1(B) m3(B) = 0.024 goes to A and B in proportion 0.20 : 0.3*0.4
    sh = pcr5_shares((B, A, B), (0.3, 0.2, 0.4))
    assert sh[A] == pytest.approx(0.015) and sh[B] == pytest.approx(0.009)


def test_MO_pcr5_vs_pcr6_partial():
    """[MO] Section 4.1: m1(A) = m2(B) = m3(B) = 0.5; PCR5 adds 0.0833 to A and 0.0416 to B, PCR6 the reverse."""
    fr = Frame("AB", model="shafer")
    A, B = fr.element("A"), fr.element("B")
    s5, s6 = pcr5_shares((A, B, B), (0.5, 0.5, 0.5)), pcr6_shares((A, B, B), (0.5, 0.5, 0.5))
    assert (round(s5[A], 4), round(s5[B], 4)) == (0.0833, 0.0417)
    assert (round(s6[A], 4), round(s6[B], 4)) == (0.0417, 0.0833)


def test_MO_five_experts_seven_classes():
    """[MO] Section 4.1 table: PCR5 decides B, PCR6 decides A."""
    fr = Frame("ABCDEFG", model="shafer")
    ms = [fr.bba({"B": 0.57, "C": 0.43})] + [fr.bba({"A": 0.58, x: 0.42}) for x in "DEFG"]
    p5 = {"A": 0.1915, "B": 0.2376, "C": 0.1542, "D": 0.1042, "E": 0.1042, "F": 0.1042, "G": 0.1042}
    p6 = {"A": 0.5138, "B": 0.1244, "C": 0.0748, "D": 0.0718, "E": 0.0718, "F": 0.0718, "G": 0.0718}
    close(fr, pcr5(fr, ms), p5, places=4)
    close(fr, pcr6(fr, ms), p6, places=4)


def test_MO_conflict_on_unions():
    """[MO] Section 3 remarks: three experts on A∪B, A∪C, B∪C; conflict 0.21 shared 7:6:5."""
    fr = Frame("ABC", model="shafer")
    ms = [fr.bba({"A|B": 0.7, "Theta": 0.3}), fr.bba({"A|C": 0.6, "Theta": 0.4}), fr.bba({"B|C": 0.5, "Theta": 0.5})]
    out = pcr6(fr, ms)
    close(fr, out, {"A": 0.21, "B": 0.14, "C": 0.09, "A|B": 0.14 + 0.21 * 7 / 18, "B|C": 0.06 + 0.21 * 5 / 18,
                    "A|C": 0.09 + 0.21 * 6 / 18, "Theta": 0.06}, places=10)
    assert fr.get(out, "A|B") == pytest.approx(0.2217, abs=5e-5) and fr.get(out, "B|C") == pytest.approx(0.1183, abs=5e-5)


def test_MO_non_associativity():
    """[MO] Section 3 remarks: experts A, B, B (categorical).  Simultaneous PCR6 gives (1/3, 2/3) and the order
    (2,3) then 1 gives (0.5, 0.5), as printed.  For the order (1,2) then 3 the paper prints (0.25, 0.75); the
    two-source formula (Eq. 8) gives m(A) = 0.5^2*1/(0.5+1) = 1/6 and m(B) = 0.5 + 1^2*0.5/(1+0.5) = 5/6, which is
    what the code returns (recorded in CHECKS_v15.txt as a discrepancy of the printed example)."""
    fr = Frame("AB", model="shafer")
    ms = [fr.bba({"A": 1.0}), fr.bba({"B": 1.0}), fr.bba({"B": 1.0})]
    close(fr, pcr6(fr, ms), {"A": 1 / 3, "B": 2 / 3}, places=12)
    close(fr, sequential(pcr6, fr, ms, [1, 2, 0]), {"A": 0.5, "B": 0.5}, places=12)
    close(fr, sequential(pcr6, fr, ms, [0, 1, 2]), {"A": 1 / 6, "B": 5 / 6}, places=12)
    close(fr, sequential(pcr5, fr, ms, [0, 1, 2]), {"A": 1 / 6, "B": 5 / 6}, places=12)


def test_pcr6_plus_DDS21_example2():
    fr = Frame("AB", model="shafer")
    ms = [fr.bba({"A": 0.6, "B": 0.1, "Theta": 0.3}), fr.bba({"A": 0.5, "B": 0.3, "Theta": 0.2}),
          fr.bba({"A": 0.4, "B": 0.1, "Theta": 0.5})]
    close(fr, pcr6(fr, ms), {"A": 0.743496, "B": 0.162245, "Theta": 0.094259})
    close(fr, pcr6_plus(fr, ms), {"A": 0.788847, "B": 0.181153, "Theta": 0.03})


@pytest.mark.parametrize("masses, model, exp_betp, exp_dsmp", [
    ({"A": 0.3, "B": 0.1, "Theta": 0.6}, "shafer", (0.6, 0.4), (0.7492, 0.2508)),          # [DS08] Example 1
    ({"A": 0.4, "Theta": 0.6}, "shafer", (0.7, 0.3), (0.9985, 0.0015)),                     # [DS08] Example 4
    ({"A&B": 0.4, "A": 0.2, "B": 0.1, "Theta": 0.3}, "free", (0.85, 0.80), (0.9990, 0.9988)),  # [DS08] Example 5
])
def test_betp_dsmp_DS08_2D(masses, model, exp_betp, exp_dsmp):
    fr = Frame("AB", model=model)
    m = fr.bba(masses)
    bp, dp = betp(fr, m), dsmp(fr, m, 0.001)
    assert (round(bp["A"], 4), round(bp["B"], 4)) == exp_betp
    # the paper prints four decimals truncated (Example 1: exact 0.7492537..., printed 0.7492)
    assert abs(dp["A"] - exp_dsmp[0]) < 1e-4 and abs(dp["B"] - exp_dsmp[1]) < 1e-4
    if masses == {"A": 0.3, "B": 0.1, "Theta": 0.6}:
        assert dp["A"] == pytest.approx(0.7492537313432835, abs=1e-12)   # reference value of the PCR6-RAG repository
    if model == "shafer":
        assert dsmp(fr, m, 0.0)["A"] == pytest.approx({0.6: 0.75, 0.7: 1.0}[exp_betp[0]])


def test_betp_dsmp_DS08_example6():
    fr = Frame("ABC", model="shafer")
    m = fr.bba({"A": 0.35, "B": 0.25, "C": 0.02, "A|B": 0.20, "A|C": 0.07, "B|C": 0.05, "Theta": 0.06})
    assert [round(betp(fr, m)[a], 4) for a in "ABC"] == [0.5050, 0.3950, 0.1000]
    assert [round(dsmp(fr, m, 0.001)[a], 4) for a in "ABC"] == [0.5665, 0.4037, 0.0298]


# --------------------------------------------------------------------------- properties
def random_bba(fr, rng, k=None):
    elems = [X for X in fr.hyper_power_set() if X]
    k = k or rng.randint(1, min(4, len(elems)))
    chosen = rng.sample(elems, k)
    w = [rng.random() + 1e-3 for _ in chosen]
    s = sum(w)
    return {X: v / s for X, v in zip(chosen, w)}


FRAMES = [Frame("AB", model="shafer"), Frame("ABC", model="shafer"), Frame("ABC", model="hybrid", empty=[("A", "B")]),
          Frame("AB")]


@pytest.mark.parametrize("fr", FRAMES, ids=repr)
def test_properties_random(fr):
    rng = random.Random(20261003)
    for _ in range(60):
        m1, m2, m3 = (random_bba(fr, rng) for _ in range(3))
        try:
            d = dempster(fr, [m1, m2])
        except TotalConflictError:
            continue
        for name, rule in RULES.items():
            out = rule(fr, [m1, m2])
            is_bba(out)
            other = rule(fr, [m2, m1])
            assert set(out) == set(other) and all(out[k] == pytest.approx(other[k], abs=1e-12) for k in out), name
            out3 = rule(fr, [m1, m2, m3]) if name != "dempster" or conflict(fr, [m1, m2, m3]) < 1 - 1e-9 else None
            if out3 is not None:
                is_bba(out3)
        # PCR5 = PCR6 for two sources
        p5, p6 = pcr5(fr, [m1, m2]), pcr6(fr, [m1, m2])
        assert set(p5) == set(p6) and all(p5[k] == pytest.approx(p6[k], abs=1e-12) for k in p5)
        # Dempster = normalised conjunctive
        c = conjunctive(fr, [m1, m2])
        K = c.pop(fr.empty, 0.0)
        assert all(d[k] == pytest.approx(v / (1 - K), abs=1e-12) for k, v in c.items())
        # Dempster is associative; simultaneous = sequential in every order
        if conflict(fr, [m1, m2, m3]) < 1 - 1e-6:
            sim = dempster(fr, [m1, m2, m3])
            for order in itertools.permutations(range(3)):
                try:
                    seq = sequential(dempster, fr, [m1, m2, m3], order)
                except TotalConflictError:
                    continue
                assert all(seq.get(k, 0) == pytest.approx(v, abs=1e-9) for k, v in sim.items())
        # vacuous BBA is neutral for Dempster, PCR5 and PCR6+
        vac = {fr.theta: 1.0}
        for rule in (dempster, pcr5, pcr6_plus):
            a, b = rule(fr, [m1, m2]), rule(fr, [m1, m2, vac])
            assert all(b.get(k, 0) == pytest.approx(v, abs=1e-12) for k, v in a.items())
        # PCR rules on the free model are the conjunctive rule (no conflict)
        if fr.model == "free":
            assert pcr6(fr, [m1, m2, m3]) == pytest.approx(conjunctive(fr, [m1, m2, m3]))


def test_yager_equals_dubois_prade_on_binary_frame():
    fr = Frame("AB", model="shafer")
    rng = random.Random(7)
    for s in (2, 3, 4):
        for _ in range(20):
            ms = [random_bba(fr, rng) for _ in range(s)]
            y, dp = yager(fr, ms), dubois_prade(fr, ms)
            assert set(y) == set(dp) and all(y[k] == pytest.approx(dp[k], abs=1e-12) for k in y)


def test_conflict_destinations_binary():
    """On {x, not_x} the rules share the conjunctive part and differ only in where K goes."""
    sh, fr = binary_frame("shafer"), binary_frame("free")
    rng = random.Random(11)
    for _ in range(30):
        ms = [random_bba(sh, rng) for _ in range(2)]
        msf = [{fr.element(sh.label(k, ascii=True)): v for k, v in m.items()} for m in ms]
        c = conjunctive(sh, ms)
        K = c.get(sh.empty, 0.0)
        assert dsmc(fr, msf).get(fr.element("x&not_x"), 0.0) == pytest.approx(K, abs=1e-12)
        y = yager(sh, ms)
        assert y.get(sh.theta, 0.0) == pytest.approx(c.get(sh.theta, 0.0) + K, abs=1e-12)
        for rule in (pcr5, pcr6):
            out = rule(sh, ms)
            assert out.get(sh.theta, 0.0) == pytest.approx(c.get(sh.theta, 0.0), abs=1e-12)
            assert out.get(sh.element("x"), 0) + out.get(sh.element("not_x"), 0) == pytest.approx(
                c.get(sh.element("x"), 0) + c.get(sh.element("not_x"), 0) + K, abs=1e-12)


def test_belief_plausibility_free_binary():
    fr = binary_frame("free")
    m = fr.bba({"x": 0.2, "not_x": 0.1, "x&not_x": 0.3, "Theta": 0.4})
    assert belief(fr, m, "x") == pytest.approx(0.5) and belief(fr, m, "not_x") == pytest.approx(0.4)
    assert plausibility(fr, m, "x") == pytest.approx(1.0)


# --------------------------------------------------------------------------- RNEL dictionary
def test_tuple_bba_roundtrip():
    tp = rnel_tuple(Reports(t=4, f=2, c=1, v=1, n=2))
    fr = binary_frame()
    m = tuple_to_bba(tp)
    assert fr.get(m, "x") == pytest.approx(tp.T) and fr.get(m, "not_x") == pytest.approx(tp.F)
    assert fr.get(m, "x&not_x") == pytest.approx(tp.C) and fr.get(m, "empty") == pytest.approx(tp.N)
    assert fr.get(m, "Theta") == pytest.approx(tp.U + tp.G)
    back = bba_to_tuple(m, u_share=tp.U / (tp.U + tp.G))
    for k, v in tp.as_dict().items():
        assert back.as_dict()[k] == pytest.approx(v)


def test_shafer_case_is_subjective_logic():
    for t, f in ((8, 0), (3, 5), (0, 0), (30, 10)):
        m = reports_to_bba(Reports(t=t, f=f), frame=binary_frame("shafer"))
        op = Opinion.from_evidence(t, f, 2.0)
        assert m == pytest.approx(opinion_to_bba(op))


def test_fused_tuple_is_not_a_bba():
    from rnel import fused_contradiction
    with pytest.raises(ValueError):
        tuple_to_bba(fused_contradiction([Reports(t=8), Reports(f=8)]))


def test_triple_bijection():
    rng = random.Random(3)
    fr = binary_frame()
    for _ in range(200):
        w = [rng.random() for _ in range(4)]
        s = sum(w)
        m = dict(zip([fr.element("x"), fr.element("not_x"), fr.element("x&not_x"), fr.theta], [v / s for v in w]))
        T, I, F = bba_to_triple(m)
        assert T + I <= 1 + 1e-12 and F + I <= 1 + 1e-12 and T + I + F >= 1 - 1e-12
        assert T + I + F - 1 == pytest.approx(m[fr.element("x&not_x")])
        back = triple_to_bba(T, I, F)
        assert all(back.get(k, 0.0) == pytest.approx(v, abs=1e-12) for k, v in m.items())
    with pytest.raises(ValueError):
        triple_to_bba(0.8, 0.3, 0.7)          # T + I > 1: outside the image


def test_belnap_least_glut_equals_glut_mass():
    pytest.importorskip("scipy")
    from rnel.neutro_credal import to_glut_frame
    fr = binary_frame()
    m = fr.bba({"x": 0.3, "not_x": 0.1, "x&not_x": 0.35, "Theta": 0.25})
    p = belnap_probability(m)
    trip = bba_to_triple(m)
    assert trip == pytest.approx((p["t"] + p["b"], p["n"], p["f"] + p["b"]))
    assert to_glut_frame(trip, "belnap").glut_lower() == pytest.approx(0.35, abs=1e-8)
    assert to_glut_frame(trip, "minimal").glut_lower() == pytest.approx(max(0.0, 0.35 - 0.25), abs=1e-8)


# --------------------------------------------------------------------------- optional cross-check
def test_crosscheck_evidencelib():
    ev = pytest.importorskip("evidencelib")
    rng = random.Random(99)
    fr = Frame("ABC", model="shafer")
    ef = ev.Frame.dst(["A", "B", "C"])

    def to_ev(m):
        return ev.MassFunction(ef, {tuple(fr.atoms[r[0]] for r in X): v for X, v in m.items()})

    for _ in range(30):
        ms = [random_bba(fr, rng) for _ in range(2)]
        e1, e2 = to_ev(ms[0]), to_ev(ms[1])
        for mine, theirs in ((pcr6(fr, ms), e1.pcr6(e2)), (yager(fr, ms), e1.yager(e2)),
                             (dubois_prade(fr, ms), e1.dubois_prade(e2))):
            for X, v in mine.items():
                assert theirs.mass([fr.atoms[r[0]] for r in X]) == pytest.approx(v, abs=1e-9)
        if conflict(fr, ms) < 1 - 1e-9:
            d = dempster(fr, ms)
            ed = e1.dempster(e2)
            for X, v in d.items():
                assert ed.mass([fr.atoms[r[0]] for r in X]) == pytest.approx(v, abs=1e-9)
        ms3 = ms + [random_bba(fr, rng)]
        e3 = to_ev(ms3[2])
        for X, v in pcr6(fr, ms3).items():
            assert e1.pcr6(e2, e3).mass([fr.atoms[r[0]] for r in X]) == pytest.approx(v, abs=1e-9)
