import numpy as np
import pytest

pytest.importorskip("sklearn")
from rnel.cluster import RNELMultiViewClusterer  # noqa: E402
from rnel.tuple import Reports, fused_contradiction  # noqa: E402


def test_vectorised_conflict_equals_definition_8_4():
    rng = np.random.default_rng(0)
    for _ in range(100):
        t, f = rng.uniform(0, 10, 4), rng.uniform(0, 10, 4)
        ref = fused_contradiction([Reports(t=a, f=b) for a, b in zip(t, f)]).C
        S = t + f + 2
        u = 2 / S
        P = t / S + 0.5 * u
        C = max(abs(P[i] - P[j]) * (1 - u[i]) * (1 - u[j]) for i in range(4) for j in range(i + 1, 4))
        assert abs(ref - C) < 1e-12


def test_partition_and_shapes():
    rng = np.random.default_rng(1)
    views = [np.vstack([rng.normal(0, 1, (60, 5)), rng.normal(6, 1, (60, 5))]) for _ in range(3)]
    m = RNELMultiViewClusterer(2, n_neighbors=10, pca_dim=3).fit(views)
    tp = m.tuples(views, training_rows=np.arange(120))
    tot = tp.T + tp.F + tp.U + tp.N + tp.G
    assert np.allclose(tot, 1)
    assert tp.features().shape == (120, 6)
