import numpy as np
import pytest

from rnel.active import ReliabilityActiveLearner, reliability


def _data(seed=0, n_per=60, k=3, V=3, d=5, bad_frac=0.2):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), n_per)
    centres = [rng.normal(0, 6, (k, d)) for _ in range(V)]
    views = [c[y] + rng.normal(0, 1, (len(y), d)) for c in centres]
    bad = rng.choice(len(y), int(bad_frac * len(y)), replace=False)
    for i in bad:
        j = rng.choice(np.where(y != y[i])[0])
        views[0][i] = views[0][j]
    return views, y, bad


@pytest.mark.parametrize("cold", ["rnel", "random", "kmeans"])
def test_exact_budget_and_prediction(cold):
    views, y, _ = _data()
    L = ReliabilityActiveLearner(3, budget=37, batch=10, cold_start=cold).fit_pool(views)
    for _ in L.queries(lambda idx: y[idx]):
        pass
    assert len(L.L_) == 37 and len(set(L.L_)) == 37
    assert (L.predict(views) == y).mean() > 0.8


def test_reliability_avoids_corrupted_objects():
    views, y, bad = _data(seed=1)
    frac = {}
    for q in ("margin", "margin_reliability"):
        L = ReliabilityActiveLearner(3, budget=60, batch=10, query=q).fit_pool(views)
        for _ in L.queries(lambda idx: y[idx]):
            pass
        frac[q] = np.isin(L.L_, bad).mean()
    assert frac["margin_reliability"] <= frac["margin"]


def test_reliability_range():
    r = reliability(np.array([0.0, 0.5, 1.0, 2.0]))
    assert r.min() >= 0 and r.max() <= 1
