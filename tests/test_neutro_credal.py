"""Tests of rnel.neutro_credal: Walley's tools applied to neutrosophic triples.

Reference values come from the dossier of the paper on neutrosophic probability and credal sets
(Theorems 1, 2, 3, 5, 8, 9; counterexamples C1-C4; Table 2)."""
import itertools

import numpy as np
import pytest

pytest.importorskip("scipy")

from rnel import neutro_credal as nc
from rnel.sl import Opinion

LP_TOL = 1e-9


# ----------------------------------------------------------------------------- helpers
def events(n):
    return [frozenset(c) for k in range(1, n) for c in itertools.combinations(range(n), k)]


def np_from_lower(L, n):
    """Dual normalised NP from a lower probability on proper events: T = L(A), F = L(not A)."""
    full = frozenset(range(n))
    return {A: (L[A], 1.0 - L[A] - L[full - A], L[full - A]) for A in L}


def key(s):
    return frozenset(int(x) for x in s.strip("[]").split(","))


EX_MEMBERS = [(0.7, 0.2, 0.1), (0.6, 0.3, 0.1), (0.1, 0.8, 0.1)]
EX_LABELS = ["A", "B", "C"]


# ----------------------------------------------------------------------------- Theorem 1: C1-C4, Table 2
def test_C1_sure_loss_with_degree_one_fifteenth():
    L = {A: (0.4 if len(A) == 1 else 0.3) for A in events(3)}
    C1 = np_from_lower(L, 3)
    assert all(abs(sum(t) - 1) < 1e-12 for t in C1.values())       # normalised
    assert not nc.avoids_sure_loss(C1, outcomes=3)
    assert nc.classify(C1, outcomes=3) == "sure_loss"
    assert abs(nc.sure_loss_degree(C1, outcomes=3) - 1 / 15) < LP_TOL
    with pytest.raises(ValueError):
        nc.natural_extension(C1, [0, 1], outcomes=3)


def test_C2_incoherent_and_correction_reduces_I():
    L = {frozenset({0}): 0.5, frozenset({1}): 0.0, frozenset({2}): 0.0,
         frozenset({0, 1}): 0.2, frozenset({0, 2}): 0.5, frozenset({1, 2}): 0.0}
    C2 = np_from_lower(L, 3)
    assert nc.avoids_sure_loss(C2, outcomes=3)
    assert not nc.is_coherent(C2, outcomes=3)
    assert nc.classify(C2, outcomes=3) == "incoherent"
    T, I, F = nc.natural_extension(C2, [0, 1], outcomes=3)
    assert abs(T - 0.5) < LP_TOL and abs(I - 0.5) < LP_TOL          # T 0.2 -> 0.5, I 0.8 -> 0.5
    corr = nc.coherent_correction(C2, outcomes=3)
    assert nc.is_coherent(corr, outcomes=3)
    for A, (T0, I0, F0) in C2.items():
        Te, Ie, Fe = corr[A]
        assert Te >= T0 - LP_TOL and Fe >= F0 - LP_TOL and Ie <= I0 + LP_TOL
        assert abs(Te + Ie + Fe - 1) < 1e-12


C3 = {"[0]": 0.09229276032189059, "[1]": 0.005206460242533165, "[2]": 0.03439367634300739,
      "[3]": 0.01224636160535773, "[0, 1]": 0.28816555423640977, "[0, 2]": 0.2842416885496426,
      "[0, 3]": 0.10453912192724832, "[1, 2]": 0.4045677922649048, "[1, 3]": 0.2479976631112588,
      "[2, 3]": 0.27331681161957216, "[0, 1, 2]": 0.65441580447154, "[0, 1, 3]": 0.5309567571051355,
      "[0, 2, 3]": 0.36560957194146276, "[1, 2, 3]": 0.7170409060061234}
C4 = {"[0]": 0.13975133220199656, "[1]": 0.026200355140305494, "[2]": 0.17955975838817476,
      "[0, 1]": 0.2453026360838626, "[0, 2]": 0.8944486961181338, "[1, 2]": 0.23019444001172373}


@pytest.mark.parametrize("L, n", [(C3, 4), (C4, 3)])
def test_C3_C4_are_coherent_and_fixed_by_the_correction(L, n):
    NP = np_from_lower({key(k): v for k, v in L.items()}, n)
    assert nc.is_coherent(NP, outcomes=n)
    corr = nc.coherent_correction(NP, outcomes=n)
    for A in NP:
        assert np.allclose(corr[A], NP[A], atol=1e-8)


def test_table2_classification_reproduced():
    """Table 2: n = 2 all coherent; n = 3: 1457 sure loss, 1505 ASL-incoherent, 38 coherent (same seed)."""
    rng = np.random.default_rng(20260929)

    def draw(n):
        L = {}
        for A in events(n):
            Ac = frozenset(range(n)) - A
            if A in L:
                continue
            x = rng.dirichlet([1, 1, 1])
            L[A], L[Ac] = x[0], x[1]
        return np_from_lower(L, n)

    for _ in range(3000):
        NP = draw(2)
        assert nc.classify(NP, outcomes=2) == "coherent"          # Theorem 1(e)
    counts = {"sure_loss": 0, "incoherent": 0, "coherent": 0}
    for _ in range(3000):
        counts[nc.classify(draw(3), outcomes=3)] += 1
    assert counts == {"sure_loss": 1457, "incoherent": 1505, "coherent": 38}


def test_correction_never_adds_indeterminacy_random():
    """Theorem 1(c) on random normalised dual assignments that avoid sure loss."""
    rng = np.random.default_rng(3)
    done = 0
    while done < 80:
        n = int(rng.integers(3, 5))
        L = {}
        for A in events(n):
            if A in L:
                continue
            x = rng.dirichlet([1, 1, 1]) * np.r_[rng.uniform(0.2, 1), rng.uniform(0.2, 1), 1]
            L[A], L[frozenset(range(n)) - A] = x[0], x[1]
        NP = np_from_lower(L, n)
        if not nc.avoids_sure_loss(NP, outcomes=n):
            continue
        corr = nc.coherent_correction(NP, outcomes=n)
        for A, (T, I, F) in NP.items():
            assert corr[A][1] <= I + LP_TOL
            assert corr[A][0] >= T - LP_TOL and corr[A][2] >= F - LP_TOL
        assert nc.is_coherent(corr, outcomes=n)
        done += 1


# ----------------------------------------------------------------------------- Theorem 2: closed form vs LP
def random_singletons(rng, n):
    x = rng.dirichlet(np.ones(3) * rng.uniform(0.3, 3), size=n)
    T = np.clip(x[:, 0] * rng.uniform(0.2, 1.5), 0, 1)
    I = np.minimum(x[:, 1], 1 - T)
    return np.c_[T, I, 1 - T - I]


def test_theorem2_closed_form_matches_lp_on_thousands_of_cases():
    rng = np.random.default_rng(7)
    stats = {"cases": 0, "coherent": 0, "asl": 0, "max_err": 0.0}
    for _ in range(3000):
        n = int(rng.integers(2, 6))
        trip = random_singletons(rng, n)
        chk = nc.singleton_intervals_check(trip)
        asl_lp = nc.sure_loss_degree({i: tuple(trip[i]) for i in range(n)}, outcomes=n) <= 1e-9
        coh_lp = nc.is_coherent(trip, method="lp")
        assert chk["avoids_sure_loss"] == asl_lp
        assert chk["coherent"] == coh_lp == nc.is_coherent(trip, method="closed")
        stats["cases"] += 1
        if asl_lp:
            stats["asl"] += 1
            stats["coherent"] += coh_lp
            for A in events(n):
                closed = nc.natural_extension(trip, A, method="closed")
                lp = nc.natural_extension(trip, A, method="lp")
                stats["max_err"] = max(stats["max_err"], max(abs(a - b) for a, b in zip(closed, lp)))
    assert stats["max_err"] < LP_TOL, stats
    assert stats["coherent"] > 100 and stats["asl"] - stats["coherent"] > 100   # both regimes exercised


def test_theorem2_reading_of_the_coherence_condition():
    # I_1 = 0.5 exceeds the deficit 0.4: not coherent, and the correction shrinks it
    trip = [(0.3, 0.5, 0.2), (0.3, 0.2, 0.5)]
    chk = nc.singleton_intervals_check(trip)
    assert abs(chk["delta"] - 0.4) < 1e-12 and chk["I_above_delta"] == [0] and not chk["coherent"]
    corr = nc.singleton_correction(trip)
    assert nc.singleton_intervals_check(corr)["coherent"]
    assert np.all(corr[:, 1] <= np.asarray(trip)[:, 1] + 1e-12)


def test_idm_triple_is_coherent_and_its_natural_extension_is_the_idm():
    rng = np.random.default_rng(11)
    for _ in range(300):
        k = int(rng.integers(2, 6))
        counts = rng.integers(0, 20, size=k).astype(float)
        s = float(rng.uniform(0.5, 4))
        N = counts.sum()
        trip = nc.idm_triple(counts, s)
        assert nc.is_coherent(trip) and nc.is_coherent(trip, method="lp")
        for A in events(k):
            nA = counts[list(A)].sum()
            T, I, F = nc.natural_extension(trip, A, method="lp")
            assert abs(T - nA / (N + s)) < LP_TOL and abs(T + I - (nA + s) / (N + s)) < LP_TOL
            assert np.allclose(nc.idm_triple(counts, s, event=A), (T, I, F), atol=LP_TOL)
    # binomial case = SL opinion with W = s (Theorem 5(b))
    op = Opinion.from_evidence(3, 5, W=2)
    assert np.allclose(nc.idm_triple([3, 5], 2)[0], (op.b, op.u, op.d))


# ----------------------------------------------------------------------------- Theorem 3: sure loss
def test_sure_loss_degree_of_one_triple():
    rng = np.random.default_rng(5)
    for _ in range(1000):
        T, I, F = rng.uniform(0, 1, 3)
        closed = nc.sure_loss_degree((T, I, F))
        assert abs(closed - max(T + F - 1, 0) / 2) < 1e-15
        lp = nc.sure_loss_degree({"A": (T, I, F), "notA": (F, I, T)}, outcomes=["A", "notA"])
        assert abs(closed - lp) < LP_TOL
        assert nc.avoids_sure_loss((T, I, F)) == (T + F <= 1)
        # I plays no role (Theorem 3(c))
        assert nc.sure_loss_degree((T, 0.0, F)) == nc.sure_loss_degree((T, 1.0, F))
    # off values (Theorem 3(d))
    assert abs(nc.sure_loss_degree((1.3, 0, 0)) - 0.3) < 1e-12
    assert abs(nc.sure_loss_degree({"A": (1.3, 0, 0)}, outcomes=["A", "B"]) - 0.3) < LP_TOL
    assert nc.avoids_sure_loss((-0.2, 0.5, 0.3)) and not nc.is_coherent((-0.2, 0.5, 0.3))


# ----------------------------------------------------------------------------- three classifiers, three classes
def test_three_classifier_example():
    l, u = np.min(EX_MEMBERS, axis=0), np.max(EX_MEMBERS, axis=0)
    trip = np.c_[l, u - l, 1 - u]
    assert np.allclose(np.c_[l, u], [[0.1, 0.7], [0.2, 0.8], [0.1, 0.1]])
    assert nc.is_coherent(trip)
    for method in ("closed", "lp"):
        T, I, F = nc.natural_extension(trip, ("A", "C"), outcomes=EX_LABELS, method=method)
        assert abs(T - 0.2) < LP_TOL and abs(T + I - 0.8) < LP_TOL          # P(A or C) in [0.2, 0.8]
    for rule in (nc.maximality, nc.e_admissible, nc.interval_dominance):
        assert rule(trip, labels=EX_LABELS) == ["A", "B"]
        assert rule(members=EX_MEMBERS, labels=EX_LABELS) == ["A", "B"]


def test_decision_rules_are_nested():
    rng = np.random.default_rng(9)
    for _ in range(200):
        K = int(rng.integers(2, 5))
        members = rng.dirichlet(np.ones(K) * 0.8, size=int(rng.integers(2, 6)))
        for kw in ({"members": members},
                   {"triples": np.c_[members.min(0), members.max(0) - members.min(0), 1 - members.max(0)]}):
            idom = set(nc.interval_dominance(**kw))
            maxi = set(nc.maximality(**kw))
            eadm = set(nc.e_admissible(**kw))
            assert eadm <= maxi <= idom and eadm


# ----------------------------------------------------------------------------- Theorems 8-9: glut frame
def test_minimal_glut_lifting_represents_the_whole_cube():
    rng = np.random.default_rng(8)
    pts = np.r_[rng.uniform(0, 1, (400, 3)), [[1, 0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 0], [0.9, 0.2, 0.8]]]
    for t in pts:
        g = nc.to_glut_frame(t)
        assert g.represents()
        assert abs(g.glut_lower() - max(t[0] + t[2] - 1, 0)) < LP_TOL
        ext = g.extreme_points()
        for k in range(3):   # the k-th extreme point attains component k
            assert abs(ext[k] @ g.event("TIF"[k]) - t[k]) < 1e-12
            assert np.all(ext[k] @ g.patterns >= t - 1e-12)
        if t[0] + t[2] > 1:   # sure loss on the classical frame, non-empty set here
            assert nc.sure_loss_degree(tuple(t)) > 0 and not g.is_empty()


def test_intermediate_frames_have_the_stated_regions():
    rng = np.random.default_rng(12)
    for t in rng.uniform(0, 1, (400, 3)):
        T, I, F = t
        assert nc.to_glut_frame(t, "belnap").represents() == (T + I <= 1 and F + I <= 1)
        assert nc.to_glut_frame(t, "disjoint").represents() == (T + I + F <= 1)


def test_n_norms_are_natural_extensions_on_the_glut_frame():
    rng = np.random.default_rng(13)
    for _ in range(300):
        x, y = rng.uniform(0, 1, 3), rng.uniform(0, 1, 3)
        lp = nc.glut_conjunction(x, y, "none", method="lp")
        assert np.allclose(lp, nc.glut_conjunction(x, y, "none"), atol=LP_TOL)
        # independence: the product of the extreme points of Theorem 8(a) attains the algebraic N-norm
        ind = nc.glut_conjunction(x, y, "independent")
        gx, gy = nc.to_glut_frame(x), nc.to_glut_frame(y)
        ex, ey = gx.extreme_points(), gy.extreme_points()
        for k, comb in enumerate((lambda a, b: a * b, lambda a, b: 1 - (1 - a) * (1 - b),
                                  lambda a, b: 1 - (1 - a) * (1 - b))):
            e = gx.event("TIF"[k])
            assert abs(comb(ex[k] @ e, ey[k] @ e) - ind[k]) < 1e-12
    assert nc.glut_conjunction((0.3, 0.5, 0.2), (0.6, 0.1, 0.4), "comonotone") == (0.3, 0.5, 0.4)


# ----------------------------------------------------------------------------- Theorem 5: SL retraction
def test_sl_retraction_and_paraconsistent_zone():
    rng = np.random.default_rng(14)
    W = 2.0
    for _ in range(1000):
        r, q = rng.uniform(0, 10, 2)
        Ts, Fs = r / (r + W), q / (q + W)
        b, d, u = nc.sl_retraction(Ts, Fs)
        op = Opinion.from_evidence(r, q, W)
        assert np.allclose((b, d, u), (op.b, op.d, op.u), atol=1e-12)
        assert (Ts + Fs > 1) == (r * q > W ** 2)


# ----------------------------------------------------------------------------- one-call diagnosis
def test_credal_diagnosis_on_the_example():
    rep = nc.credal_diagnosis(EX_MEMBERS, labels=EX_LABELS, events={"A or C": ["A", "C"]})
    assert rep["status"] == "coherent" and rep["decision"] == ["A", "B"]
    T, I, F = rep["events"]["A or C"]
    assert abs(T - 0.2) < 1e-12 and abs(1 - F - 0.8) < 1e-12
    for k in EX_LABELS:
        s = rep["rnel"]["per_class"][k]
        assert abs(s["a"] + s["b"] + s["u"] + s["c"] - 1) < 1e-12
    # the classifiers disagree about A: between-source conflict; about C they agree
    assert rep["rnel"]["per_class"]["A"]["c"] > 0.2 and rep["rnel"]["per_class"]["C"]["c"] == 0
    assert rep["rnel"]["C_star"] > 0
    for rule in ("interval_dominance", "e_admissible"):
        for cs in ("hull", "intervals"):
            assert nc.credal_diagnosis(EX_MEMBERS, labels=EX_LABELS, rule=rule, credal_set=cs)["decision"] == ["A", "B"]


def test_credal_diagnosis_with_a_credal_ensemble():
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification

    from rnel.credal import CredalEnsemble

    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_classes=3, random_state=0)
    ens = CredalEnsemble(random_state=0).fit(X[:250], y[:250])
    reps = nc.credal_diagnosis(ensemble=ens, X=X[250:260], events={"0 or 1": [0, 1]})
    assert len(reps) == 10
    E = ens.evidence(X[250:260], "leaf")
    for i, rep in enumerate(reps):
        assert rep["status"] in ("coherent", "incoherent")       # a hull of genuine distributions avoids sure loss
        assert rep["decision"] and set(rep["decision"]) <= set(ens.classes_)
        lo, hi = ens.bounds(X[250 + i:251 + i])
        assert np.allclose([rep["triples"][c][0] for c in ens.classes_], lo[0])
        # pooled leaf evidence: u = W / (total + W)
        u = rep["rnel"]["per_class"][ens.classes_[0]]["u"]
        assert abs(u - 2.0 / (E[i].sum() + 2.0)) < 1e-9
