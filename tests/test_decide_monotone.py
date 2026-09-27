"""The decision policy must respect AGM inclusion: a report for x never produces the answer 'refuted'
(and symmetrically). v0.1.0 violated this at (t=0, f=3) + 1 supporting report with the default policy."""
import itertools

import pytest

from rnel import Reports, fused_contradiction, rnel_tuple
from rnel.decide import Policy


def _violations(pol, rng=30):
    out = []
    for t, f in itertools.product(range(rng), repeat=2):
        a = pol.decide(rnel_tuple(Reports(t=t, f=f)))[0]
        if a != "F" and pol.decide(rnel_tuple(Reports(t=t + 1, f=f)))[0] == "F":
            out.append(("+t", t, f))
        if a != "T" and pol.decide(rnel_tuple(Reports(t=t, f=f + 1)))[0] == "T":
            out.append(("+f", t, f))
    return out


def test_legacy_policy_shows_the_failure():
    legacy = Policy(side_gate=False)
    assert legacy.decide(rnel_tuple(Reports(t=0, f=3)))[0] == "G"
    assert legacy.decide(rnel_tuple(Reports(t=1, f=3)))[0] == "F"
    assert ("+t", 0, 3) in _violations(legacy)


@pytest.mark.parametrize("G", [0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
@pytest.mark.parametrize("margin", [0.05, 0.15, 0.3])
def test_side_gate_is_monotone(G, margin):
    assert _violations(Policy(G=G, margin=margin)) == []


def test_side_gate_agrees_with_legacy_when_evidence_is_one_sided():
    new, old = Policy(), Policy(side_gate=False)
    for k in range(30):
        for r in (Reports(t=k), Reports(f=k)):
            assert new.decide(rnel_tuple(r)) == old.decide(rnel_tuple(r))


def test_fused_sources_case():
    pol = Policy()
    x0 = fused_contradiction([Reports(f=1)] * 3)
    x1 = fused_contradiction([Reports(f=1)] * 3 + [Reports(t=1)])
    assert pol.decide(x0)[0] == "G"
    assert pol.decide(x1)[0] != "F"


def test_readme_example_unchanged():
    x = fused_contradiction([Reports(t=10), Reports(f=10)])
    assert Policy().decide(x)[0] == "C"
