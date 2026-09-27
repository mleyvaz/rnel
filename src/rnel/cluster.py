"""RNEL multi-view clustering (requires numpy and scikit-learn).

Each view is a source. Clusters are found on all views jointly; then, for every point, each view reports
evidence about which cluster the point belongs to, from the cluster labels of its nearest neighbours in
that view, scaled by how dense the neighbourhood is. The reports are typed as in Definition 8.1:

  t  support for the assigned cluster (evidence of every view for it)
  c  contradiction: confident views pointing to different clusters (Definition 8.4, between sources)
  u  undetermined: within a view, evidence split between the two best clusters (overlap)
  n  neither: the point is far from every cluster in a view (outlier), counted as evidence for "none"
  W  prior weight; G = W / (t + c + u + n + W) is the ignorance

and the tuple is (T, C, U, N, G) = (t, c, u, n, W) / S. With one view, no outliers and well separated
clusters it reduces to the support of the assigned cluster and the ignorance, as in Subjective Logic.
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
    C: np.ndarray
    U: np.ndarray
    N: np.ndarray
    G: np.ndarray

    def dominant(self) -> np.ndarray:
        """Name of the largest component per point."""
        M = np.stack([self.T, self.C, self.U, self.N, self.G], 1)
        return np.array(["T", "C", "U", "N", "G"])[M.argmax(1)]


class RNELMultiViewClusterer:
    def __init__(self, n_clusters: int, n_neighbors: int = 20, W: float = 2.0, pca_dim: int = 20,
                 random_state: int = 0):
        self.k, self.m, self.W, self.pca_dim, self.rs = n_clusters, n_neighbors, W, pca_dim, random_state

    # ------------------------------------------------------------------ fitting
    def _embed(self, v, X, fit):
        if fit:
            sc = StandardScaler().fit(X)
            d = min(self.pca_dim, X.shape[1], X.shape[0] - 1)
            pca = PCA(d, random_state=self.rs).fit(sc.transform(X))
            self.scalers_.append(sc)
            self.pcas_.append(pca)
        return self.pcas_[v].transform(self.scalers_[v].transform(X))

    def fit(self, views: list[np.ndarray]) -> "RNELMultiViewClusterer":
        self.scalers_, self.pcas_ = [], []
        Z = [self._embed(v, X, True) for v, X in enumerate(views)]
        joint = np.hstack([z / np.sqrt(z.shape[1]) for z in Z])  # each view weighted equally
        self.kmeans_ = KMeans(self.k, n_init=10, random_state=self.rs).fit(joint)
        self.labels_ = self.kmeans_.labels_
        self.nn_, self.rbar_ = [], []
        for z in Z:
            nn = NearestNeighbors(n_neighbors=self.m + 1).fit(z)
            dist, _ = nn.kneighbors(z)
            self.nn_.append(nn)
            self.rbar_.append(np.median(dist[:, -1]))  # typical distance to the m-th neighbour
        self.Ztrain_ = Z
        return self

    # ------------------------------------------------------------------ evidence
    def view_evidence(self, views: list[np.ndarray], exclude_self: bool = False):
        """Per view: evidence counts over clusters (n_points, k) and density factor rho in (0, 1]."""
        out = []
        for v, X in enumerate(views):
            z = self._embed(v, X, False)
            dist, idx = self.nn_[v].kneighbors(z, n_neighbors=self.m + (1 if exclude_self else 0))
            if exclude_self:
                dist, idx = dist[:, 1:], idx[:, 1:]
            lab = self.labels_[idx]
            counts = np.stack([(lab == c).sum(1) for c in range(self.k)], 1).astype(float)
            rho = np.minimum(1.0, self.rbar_[v] / np.maximum(dist[:, -1], 1e-12))
            out.append((counts * rho[:, None], rho))
        return out

    def tuples(self, views: list[np.ndarray], exclude_self: bool = False) -> ClusterTuples:
        ev = self.view_evidence(views, exclude_self)
        E = np.stack([e for e, _ in ev], 0)                     # (views, n, k)
        rho = np.stack([r for _, r in ev], 0)                   # (views, n)
        total = E.sum(0)                                        # fused evidence over clusters
        assigned = total.argmax(1)
        n_pts = E.shape[1]
        order = np.argsort(-E, axis=2)
        top = np.take_along_axis(E, order[..., :1], 2)[..., 0]  # (views, n)
        second = np.take_along_axis(E, order[..., 1:2], 2)[..., 0]
        top_lab = order[..., 0]
        # t: support of every view for the assigned cluster, minus what is counted as undetermined
        support = np.take_along_axis(E, assigned[None, :, None].repeat(E.shape[0], 0), 2)[..., 0]
        u = second.sum(0)                                        # within-view split (overlap)
        t = np.maximum(support.sum(0) - u, 0.0)
        # c: largest pairwise conflict between views that confidently favour different clusters
        c = np.zeros(n_pts)
        V = E.shape[0]
        for a in range(V):
            for b in range(a + 1, V):
                diff = top_lab[a] != top_lab[b]
                c = np.maximum(c, np.where(diff, np.minimum(top[a] - second[a], top[b] - second[b]), 0.0))
        n = (self.m * (1 - rho)).sum(0)                          # missing neighbourhood mass: "none"
        S = t + c + u + n + self.W
        return ClusterTuples(assigned, t / S, c / S, u / S, n / S, self.W / S)
