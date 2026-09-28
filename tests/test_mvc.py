import numpy as np

from rnel.mvc import RNELMVC, type_points


def _blobs(seed=0, n_per=60, k=3, V=3, d=5):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), n_per)
    centres = [rng.normal(0, 6, (k, d)) for _ in range(V)]
    views = [c[y] + rng.normal(0, 1, (len(y), d)) for c in centres]
    return views, y


def test_partition_and_shapes():
    views, y = _blobs()
    r = RNELMVC(3, n_neighbors=10).fit(views)
    assert r.labels.shape == y.shape and r.view_weights.shape == (3,)
    assert np.isclose(r.view_weights.sum(), 1)
    tot = r.T + r.F + r.U + r.N + r.G
    assert np.allclose(tot, 1)
    assert np.all((r.C >= 0) & (r.C < 1))


def test_G_is_constant_with_fixed_neighbourhoods():
    # S = V m + W for every point (disclosed in the paper): G carries no per-point information here
    views, _ = _blobs()
    r = RNELMVC(3, n_neighbors=10).fit(views)
    assert np.allclose(r.G, r.G[0])
    assert np.isclose(r.G[0], 2.0 / (3 * 10 + 2.0))


def test_noise_view_is_down_weighted():
    views, y = _blobs(seed=1)
    rng = np.random.default_rng(5)
    views[1] = rng.normal(0, 1, views[1].shape)
    r = RNELMVC(3, n_neighbors=10).fit(views)
    assert r.view_weights[1] < 1 / 3


def test_swapped_views_get_low_weight_and_high_conflict():
    views, y = _blobs(seed=2)
    rng = np.random.default_rng(3)
    bad = rng.choice(len(y), 15, replace=False)
    for i in bad:
        j = rng.choice(np.where(y != y[i])[0])
        views[0][i] = views[0][j]
        views[1][i] = views[1][j]
    r = RNELMVC(3, n_neighbors=10).fit(views)
    good = np.setdiff1d(np.arange(len(y)), bad)
    assert r.point_weights[bad].mean() < r.point_weights[good].mean()
    assert r.C[bad].mean() > r.C[good].mean()


def test_typing_labels():
    views, _ = _blobs()
    t = type_points(RNELMVC(3, n_neighbors=10).fit(views))
    assert set(t) <= {"clean", "conflict", "ambiguous", "outlier"}


def test_convergence_flag_and_learning_conflict():
    views, _ = _blobs()
    m = RNELMVC(3, n_neighbors=10, max_iter=1)
    m.fit(views)
    assert m.converged_ in (True, False)
    m2 = RNELMVC(3, n_neighbors=10)
    r = m2.fit(views)
    assert m2.converged_ is True
    assert np.allclose(r.point_weights, m2.support_share_ * (1 - m2.C_learn_))
