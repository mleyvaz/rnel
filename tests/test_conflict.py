"""Tests of rnel.conflict (experimental): the results of the conflict dossier on random profiles."""
import itertools
import random

import pytest

from rnel import Opinion, Reports
from rnel import conflict as cf
from rnel import sl

TOL = 1e-9


def rand_profile(rng, nmax=6):
    n = rng.randint(1, nmax)
    return [(rng.expovariate(0.2) * (rng.random() < 0.8), rng.expovariate(0.2) * (rng.random() < 0.8))
            for _ in range(n)]


def test_three_forms_of_k_between_agree():
    rng = random.Random(1)
    for _ in range(3000):
        P = rand_profile(rng)
        R, S = sum(r for r, _ in P), sum(s for _, s in P)
        kb = cf.k_between(P)
        assert abs(kb - (min(R, S) - cf.k_within(P))) < TOL
        assert abs(kb - cf.sup_gap([list(e) for e in P])) < TOL
        assert kb >= -TOL


def test_order_and_bracketing_invariance():
    rng = random.Random(2)
    for _ in range(500):
        P = rand_profile(rng, 5)
        ref = cf.c_star(P)
        for perm in itertools.permutations(P):
            st = cf.ConflictState.of(perm[0])
            for x in perm[1:]:
                st = st + cf.ConflictState.of(x)
            assert abs(st.c_star() - ref) < TOL
        if len(P) >= 3:  # right bracketing
            st = cf.ConflictState.of_profile(P[1:])
            assert abs((cf.ConflictState.of(P[0]) + st).c_star() - ref) < TOL


def test_definition_8_4_sequential_is_order_dependent():
    A, B, D = (10, 0), (0, 10), (10, 0)
    assert round(cf.def84_sequential([A, B, D]), 3) == 0.579
    assert round(cf.def84_sequential([A, D, B]), 3) == 0.660
    assert cf.c_star([A, B, D]) == cf.c_star([A, D, B])


def test_axioms_characterising_k_between():
    rng = random.Random(3)
    for _ in range(2000):
        P = rand_profile(rng)
        kb = cf.k_between(P)
        i, d = rng.randrange(len(P)), rng.random() * 5
        Q = list(P)
        Q[i] = (P[i][0] + d, P[i][1] + d)                                    # B1 internal cancellation
        assert abs(cf.k_between(Q) - kb) < TOL
        assert abs(cf.k_between(P + [(d, 0), (0, d)]) - kb - d) < TOL        # B4 opposed pair
        pro = [e for e in P if e[0] >= e[1]]                                  # B2 consolidation of allies
        if len(pro) >= 2:
            rest = [e for e in P if e[0] < e[1]]
            merged = (sum(r for r, _ in pro), sum(s for _, s in pro))
            assert abs(cf.k_between(rest + [merged]) - kb) < TOL
        assert cf.k_between([(r, 0) for r, _ in P]) == 0                     # B3 unanimity


def test_credal_gap_is_proportional_to_k_between():
    rng = random.Random(4)
    for _ in range(3000):
        P = rand_profile(rng)
        lo, hi = sorted((rng.random(), rng.random()))
        assert abs(cf.credal_gap(P, lo, hi) - (hi - lo) * cf.k_between(P)) < 1e-8


def test_identifiability_from_the_fused_tuple():
    rng = random.Random(5)
    for _ in range(2000):
        P = rand_profile(rng)
        st = cf.ConflictState.of_profile(P)
        t = st.tuple(2.0)
        back = cf.state_from_tuple(t["T"], t["F"], t["G"], t["C"], 2.0)
        assert abs(back.R - st.R) < 1e-7 and abs(back.S - st.S) < 1e-7 and abs(back.Kb - st.Kb) < 1e-7


def test_sl_projection_is_unchanged():
    rng = random.Random(6)
    for _ in range(500):
        P = rand_profile(rng)
        op = cf.ConflictState.of_profile(P).opinion()
        fused = Opinion.from_evidence(*P[0])
        for x in P[1:]:
            fused = sl.cumulative_fusion(fused, Opinion.from_evidence(*x))
        assert abs(op.b - fused.b) < TOL and abs(op.d - fused.d) < TOL


def test_bounds_of_c_star():
    rng = random.Random(7)
    for _ in range(2000):
        st = cf.ConflictState.of_profile(rand_profile(rng))
        t = st.tuple()
        assert -TOL <= t["C"] <= 2 * min(t["T"], t["F"]) + TOL <= 1 - t["G"] + TOL


def test_dc_against_consensus_and_opposition():
    # DC is positive between two sources that only support x (violates the unanimity axiom)
    assert sl.degree_of_conflict(Opinion.from_evidence(1, 0), Opinion.from_evidence(100, 0)) > 0.1
    assert cf.k_between([(1, 0), (100, 0)]) == 0
    # with base rate a = 0.1, DC vanishes between sources that lean opposite ways
    A, B = Opinion.from_evidence(1, 0, a=0.1), Opinion.from_evidence(10, 13.5, a=0.1)
    assert sl.degree_of_conflict(A, B) < 1e-12 and cf.k_between([(1, 0), (10, 13.5)]) == 1


def test_duplication():
    A, B = (10.0, 0.0), (0.0, 10.0)
    reps = [("a", A), ("b", B)]
    base = cf.dependence_interval(reps)["cautious"]
    for k in (1, 2, 5):
        dup = reps + [("a", A)] * k
        d = cf.dependence_interval(dup)
        assert d["cautious"] == base                        # provenance-aware: copies add nothing
        assert d["cumulative"]["T"] > base["T"]             # cumulative: copies inflate T
        assert d["cumulative"]["C"] < base["C"]             # ... and dilute the conflict
        assert 0 < d["I_dep"] < 1


def test_atom_union_and_frechet_bounds():
    rng = random.Random(8)
    for _ in range(500):
        pool = list(range(30))
        reps = [{a: 1 for a in rng.sample(pool, rng.randint(0, 12))} for _ in range(rng.randint(1, 4))]
        true_r, _ = cf.atom_union(reps)
        sizes = [len(x) for x in reps]
        assert max(sizes) <= true_r <= sum(sizes)
        lab = [("g", (float(s), 0.0)) for s in sizes]
        assert cf.group_counts(lab, "cautious")[0][0] <= true_r
        assert cf.group_counts(lab, "average")[0][0] <= cf.group_counts(lab, "cautious")[0][0]


def test_reports_are_accepted():
    assert cf.k_between([Reports(t=3, f=1), Reports(t=0, f=4)]) == 2
