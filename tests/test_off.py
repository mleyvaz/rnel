import warnings

import numpy as np
import pytest

from rnel.off import EvidenceLedger, OffOpinion, OffValueWarning, cumulative_unfusion, off_from_evidence
from rnel.sl import Opinion, cumulative_fusion


def test_ledger_without_retraction_is_sl():
    L = EvidenceLedger().report("A", 8, 2).report("B", 3, 1)
    assert L.opinion() == Opinion.from_evidence(11, 3)


def test_retracting_a_source_equals_cumulative_unfusion():
    A, B = Opinion.from_evidence(8, 2), Opinion.from_evidence(3, 1)
    fused = cumulative_fusion(A, B)
    L = EvidenceLedger().report("A", 8, 2).report("B", 3, 1).retract_source("B")
    un = cumulative_unfusion(fused, B)
    assert np.allclose([L.opinion().b, L.opinion().d, L.opinion().u], [A.b, A.d, A.u])
    assert np.allclose([un.b, un.d, un.u], [A.b, A.d, A.u])


def test_partial_retraction_fake_reviews():
    # 40 positive reviews, 25 of them found fake and removed; 10 negative reviews
    L = EvidenceLedger().report("shop", 40, 10).retract("shop", for_x=25)
    assert L.net() == (15, 10)
    assert L.opinion() == Opinion.from_evidence(15, 10)


def test_inverted_source_is_positive_evidence_on_the_complement():
    # a compromised relay reports 9 "true" and 1 "false"; known to be inverted
    L = EvidenceLedger().report("relay", 9, 1).invert("relay")
    o = L.opinion()
    assert o == Opinion.from_evidence(1, 9)
    assert 0 <= o.projected <= 1


def test_over_retraction_is_refused_or_flagged():
    L = EvidenceLedger().report("A", 2, 0).retract("A", for_x=3)  # net r = -1, S = 1
    with pytest.raises(ValueError):
        L.opinion()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        off = L.opinion(allow_off=True)
    assert isinstance(off, OffOpinion) and any(issubclass(x.category, OffValueWarning) for x in w)
    assert off.b < 0 and not off.in_simplex
    with pytest.raises(ValueError):
        off.to_opinion()


def test_beta_admissibility_boundary():
    W, a = 2.0, 0.5
    assert off_from_evidence(-0.5, 0, W, a).beta_admissible        # alpha = -0.5 + 1 > 0
    assert not off_from_evidence(-1.5, 0, W, a).beta_admissible    # alpha = -0.5 < 0: no Beta density


def test_unfusion_rejects_foreign_opinion():
    with pytest.raises(ValueError):
        cumulative_unfusion(Opinion.from_evidence(2, 1), Opinion.from_evidence(5, 0))


def test_section10_relay_and_complement_alternative():
    from rnel.off import complement_discount, typed_discount
    y = typed_discount(-0.2, {"T": 0.8, "F": 0.1, "U_evidence": 0.1}, "U_platform")
    P_off = y["T"] + 0.5 * (1 - y["T"] - y["F"])
    b, d, u = complement_discount(-0.2, 0.8, 0.1, 0.1)
    assert abs(P_off - 0.43) < 1e-9 and abs(b + 0.5 * u - 0.43) < 1e-9
    assert min(b, d, u) >= 0 and y["T"] < 0


def test_section10_nu_claims():
    from rnel.off import retraction_nu
    b, d, u, _ = retraction_nu(0.10, 0.02, 0.05)       # sigma < 1: nu is NOT the SL opinion
    o = Opinion.from_evidence(10, 5)
    assert abs(b - o.b) > 0.1
    b, d, u, _ = retraction_nu(1.0, 0.2, 0.5)          # sigma > 1: it is
    assert abs(b - o.b) < 1e-9


def test_check_script_runs():
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "check_off_extensions.py"), run_name="x")
