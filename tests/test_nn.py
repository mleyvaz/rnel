import pytest

torch = pytest.importorskip("torch")

from rnel.nn import (PerSourceEvidential, RNELHead, fused_dissonance, rnel_from_evidence,  # noqa: E402
                     source_conflict, typed_evidential_loss)


def test_tuple_sums_to_one_and_reduces_to_sl():
    e = torch.tensor([[7.0, 3.0, 0.0, 0.0, 0.0]])
    t = rnel_from_evidence(e, W=2.0)
    assert torch.allclose(t.sum(-1), torch.ones(1))
    assert torch.allclose(t[0, [0, 1, 5]], torch.tensor([7 / 12, 3 / 12, 2 / 12]))


def test_head_and_loss():
    torch.manual_seed(0)
    head = RNELHead(8)
    e = head(torch.randn(4, 8))
    assert (e >= 0).all()
    loss = typed_evidential_loss(e, torch.tensor([0, 1, 2, 4]), reg=0.1)
    assert torch.isfinite(loss)


def test_source_conflict_detects_disagreement_that_fusion_hides():
    mask = torch.ones(2, 2)
    disagree = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])   # two confident sources, opposite sides
    balanced = torch.tensor([[[5.0, 5.0], [5.0, 5.0]]])     # two balanced sources
    e = torch.cat([disagree, balanced])
    E = e.mean(1)
    assert torch.allclose(fused_dissonance(E)[0], fused_dissonance(E)[1])  # fusion: identical
    c = source_conflict(e, mask)
    assert c[0] > c[1]                                                        # sources: separated


def test_per_source_model_shapes():
    m = PerSourceEvidential(6, hidden=16)
    x, mask = torch.randn(3, 4, 6), torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1.0]])
    E, e = m(x, mask)
    assert E.shape == (3, 2) and e.shape == (3, 4, 2) and (e[0, 3] == 0).all()
