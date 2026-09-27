"""RNEL multi-view clustering (requires numpy and scikit-learn).

Each view is a source of reports about whether a point belongs to its assigned cluster a. In view v the
m nearest neighbours of the point (in that view) are counted by cluster, and each count is scaled by a
density factor rho_v in (0, 1] (1 when the neighbourhood is as dense as usual). The evidence of view v is
split, without double counting, into report counts of Definition 8.1:

  support    t_v = c_v(a) - min(c_v(a), o_v)          neighbours in the assigned cluster
  against    f_v = o_v - min(c_v(a), o_v)             neighbours in other clusters, o_v = sum_{k!=a} c_v(k)
  undetermined  v_v = 2 min(c_v(a), o_v)              the balanced part of a split neighbourhood
  neither    n_v = m (1 - rho_v)                      neighbourhood mass missing because the point is remote

so that t_v + f_v + v_v = c_v(a) + o_v. The views are fused with rnel.tuple.fused_contradiction
(Definition 8.4): T, F, U, N and G come from the summed counts, and C is the largest pairwise Subjective
Logic degree of conflict between the views' opinions (t_v, f_v) about membership in a. As in Definition 8.4
the total may exceed 1 when views conflict.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterTuples:
    labels: np.ndarray
    T: np.ndarray
    F: np.ndarray
    C: np.ndarray
    U: np.ndarray
    N: np.ndarray
    G: np.ndarray

    def features(self) -> np.ndarray:
        return np.stack([self.T, self.F, self.C, self.U, self.N, self.G], 1)


class RNELMultiViewClusterer:
    def __init__(self, n_clusters: int, n_neighbors: int = 20, W: float = 2.0, pca_dim: int = 20,
                 base_rate: float = 0.5, random_state: int = 0):
        self.k, self.m, self.W, self.pca_dim, self.a, self.rs = (n_clusters, n_neighbors, W, pca_dim,
                                                                  base_rate, random_state)

    def _embed(self, v, X, fit):
        if fit:
            sc = StandardScaler().fit(X)
            d = min(self.pca_dim, X.shape[1], X.shape[0] - 1)
            self.scalers_.append(sc)
            self.pcas_.append(PCA(d, random_state=self.rs).fit(sc.transform(X)))
        return self.pcas_[v].transform(self.scalers_[v].transform(X))

    def embed(self, views):
        return [self._embed(v, X, False) for v, X in enumerate(views)]

    def fit(self, views: list[np.ndarray]) -> "RNELMultiViewClusterer":
        self.scalers_, self.pcas_ = [], []
        Z = [self._embed(v, X, True) for v, X in enumerate(views)]
        joint = np.hstack([z / np.sqrt(z.shape[1]) for z in Z])
        self.kmeans_ = KMeans(self.k, n_init=10, random_state=self.rs).fit(joint)
        self.labels_ = self.kmeans_.labels_
        self.nn_, self.rbar_ = [], []
        for z in Z:
            nn = NearestNeighbors(n_neighbors=self.m + 1).fit(z)
            dist, _ = nn.kneighbors(z)                 # first neighbour is the point itself
            self.nn_.append(nn)
            self.rbar_.append(np.median(dist[:, -1]))  # typical distance to the m-th other point
        return self

    def neighbours(self, views, training_rows: np.ndarray | None = None):
        """Per view (dist, idx) of the m nearest fitting points. If the queries are fitting points, pass
        their row indices in training_rows so that each point is removed from its own neighbourhood."""
        out = []
        for v, z in enumerate(self.embed(views)):
            extra = 1 if training_rows is not None else 0
            dist, idx = self.nn_[v].kneighbors(z, n_neighbors=self.m + extra)
            if training_rows is not None:
                keep = idx != training_rows[:, None]
                # drop the query's own row (or the last neighbour if the row is absent)
                sel = np.argsort(~keep, axis=1, kind="stable")[:, :self.m]
                dist, idx = np.take_along_axis(dist, sel, 1), np.take_along_axis(idx, sel, 1)
            out.append((dist, idx))
        return out

    def view_counts(self, views, training_rows=None):
        """Per view: density-scaled neighbour counts by cluster (n, k) and the density factor rho (n,)."""
        res = []
        for v, (dist, idx) in enumerate(self.neighbours(views, training_rows)):
            lab = self.labels_[idx]
            counts = np.stack([(lab == c).sum(1) for c in range(self.k)], 1).astype(float)
            rho = np.minimum(1.0, self.rbar_[v] / np.maximum(dist[:, -1], 1e-12))
            res.append((counts * rho[:, None], rho, dist[:, -1] / self.rbar_[v]))
        return res

    def tuples(self, views, training_rows=None) -> ClusterTuples:
        vc = self.view_counts(views, training_rows)
        E = np.stack([c for c, _, _ in vc])                   # (V, n, k)
        rho = np.stack([r for _, r, _ in vc])                 # (V, n)
        assigned = E.sum(0).argmax(1)
        sup = np.take_along_axis(E, np.broadcast_to(assigned[None, :, None], E.shape[:2] + (1,)), 2)[..., 0]
        oth = E.sum(2) - sup
        mn = np.minimum(sup, oth)
        t, f, u = sup - mn, oth - mn, 2 * mn                   # (V, n) report counts per view
        nn = self.m * (1 - rho)
        S = t.sum(0) + f.sum(0) + u.sum(0) + nn.sum(0) + self.W
        # Definition 8.4: largest pairwise SL degree of conflict between view opinions (t_v, f_v)
        Sv = t + f + self.W
        b, d, uu = t / Sv, f / Sv, self.W / Sv
        P = b + self.a * uu
        V = E.shape[0]
        C = np.zeros(E.shape[1])
        for i in range(V):
            for j in range(i + 1, V):
                C = np.maximum(C, np.abs(P[i] - P[j]) * (1 - uu[i]) * (1 - uu[j]))
        return ClusterTuples(assigned, t.sum(0) / S, f.sum(0) / S, C, u.sum(0) / S, nn.sum(0) / S, self.W / S)
