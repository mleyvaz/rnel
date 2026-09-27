"""RNEL-MVC: multi-view clustering in which the RNEL tuple of every point enters the learning loop.

Each iteration
  1. computes, for every point and view, neighbour evidence about membership in the point's current cluster
     (split into support t, against f, undetermined u and neither n, as in rnel.cluster) and the RNEL tuple
     (T, F, C, U, N, G), where C is the largest pairwise Subjective Logic degree of conflict between views;
  2. sets a point weight  pi_i = (share of the point's neighbour evidence that supports its cluster) x (1 - C_i),
     where C_i is the Definition 8.4 conflict computed from the unsplit per-view opinions (support s_v,
     opposition o_v). Conflicting, ambiguous and remote points therefore move the centroids less (the role of
     the doubt and empty-set masses in ECM). weight_evidence='tuple' uses T/(1-G)(1-C) instead (ablation);
  3. sets a view weight  w_v proportional to the pi-weighted mean projected probability that view v gives to
     the points' clusters: a view that systematically disagrees with the consensus loses weight;
  4. updates per-view centroids as pi-weighted means and reassigns each point to the cluster minimising the
     w-weighted sum of per-view squared distances (each view scaled by its mean within-cluster dispersion).
It stops when the labels no longer change. The output is a partition, view weights, the tuple of every point
and, with type_points(), a typed credal-style reading: clean, conflict (views disagree), ambiguous
(meta-cluster {a, b}) or outlier (empty set).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class MVCResult:
    labels: np.ndarray
    runner_up: np.ndarray
    view_weights: np.ndarray
    point_weights: np.ndarray
    T: np.ndarray
    F: np.ndarray
    C: np.ndarray
    U: np.ndarray
    N: np.ndarray
    G: np.ndarray
    n_iter: int

    def features(self) -> np.ndarray:
        return np.stack([self.T, self.F, self.C, self.U, self.N, self.G], 1)


def embed_views(views, pca_dim=20, random_state=0):
    out = []
    for X in views:
        Z = StandardScaler().fit_transform(X)
        d = min(pca_dim, Z.shape[1], Z.shape[0] - 1)
        out.append(PCA(d, random_state=random_state).fit_transform(Z))
    return out


class RNELMVC:
    def __init__(self, n_clusters, n_neighbors=20, W=2.0, base_rate=0.5, max_iter=30, point_weights=True,
                 view_weights=True, split=True, weight_evidence="raw", use_conflict=True, random_state=0):
        self.k, self.m, self.W, self.a = n_clusters, n_neighbors, W, base_rate
        self.max_iter, self.use_pi, self.use_w, self.split, self.rs = (max_iter, point_weights, view_weights,
                                                                       split, random_state)
        self.use_conflict = use_conflict  # ablation: drop the (1 - C) factor from the point weights
        self.weight_evidence = weight_evidence  # 'raw': weights from unsplit counts; 'tuple': from (T, G, C)

    # -------------------------------------------------------------- evidence
    def _neighbourhoods(self, Z):
        self.idx_, self.rho_ = [], []
        for z in Z:
            dist, idx = NearestNeighbors(n_neighbors=self.m + 1).fit(z).kneighbors(z)
            dist, idx = dist[:, 1:], idx[:, 1:]              # drop the point itself
            rbar = np.median(dist[:, -1])
            self.idx_.append(idx)
            self.rho_.append(np.minimum(1.0, rbar / np.maximum(dist[:, -1], 1e-12)))

    def _tuple(self, labels):
        n, V = len(labels), len(self.idx_)
        E = np.stack([np.stack([(labels[idx] == c).sum(1) for c in range(self.k)], 1) * rho[:, None]
                      for idx, rho in zip(self.idx_, self.rho_)]).astype(float)       # (V, n, k)
        s = E[:, np.arange(n), labels]                                                   # (V, n)
        o = E.sum(2) - s
        if self.split:
            mu = np.minimum(s, o)
            t, f, u = s - mu, o - mu, 2 * mu
        else:
            t, f, u = s, o, np.zeros_like(s)
        nn = self.m * (1 - np.stack(self.rho_))
        S = t.sum(0) + f.sum(0) + u.sum(0) + nn.sum(0) + self.W
        Sv = t + f + self.W
        b, eps = t / Sv, self.W / Sv
        P = b + self.a * eps                                                             # (V, n)
        C = np.zeros(n)
        for i in range(V):
            for j in range(i + 1, V):
                C = np.maximum(C, np.abs(P[i] - P[j]) * (1 - eps[i]) * (1 - eps[j]))
        tot = E.sum(0).copy()
        tot[np.arange(n), labels] = -np.inf
        runner = tot.argmax(1)
        G = self.W / S
        # unsplit evidence (support s, opposition o) for the learning weights
        Sr = s + o + self.W
        Pr = s / Sr + self.a * self.W / Sr
        er = self.W / Sr
        Cr = np.zeros(n)
        for i in range(V):
            for j in range(i + 1, V):
                Cr = np.maximum(Cr, np.abs(Pr[i] - Pr[j]) * (1 - er[i]) * (1 - er[j]))
        support_share = s.sum(0) / (s.sum(0) + o.sum(0) + nn.sum(0))
        return dict(Pr=Pr, pi_raw=support_share * ((1 - Cr) if self.use_conflict else 1.0), T=t.sum(0) / S, F=f.sum(0) / S, C=C, U=u.sum(0) / S, N=nn.sum(0) / S, G=G, P=P,
                    runner=runner)

    def _pi(self, tp):
        if self.weight_evidence == "raw":
            return tp["pi_raw"]
        return tp["T"] / (1 - tp["G"]) * (1 - tp["C"])

    # -------------------------------------------------------------- learning
    def fit(self, views) -> MVCResult:
        Z = embed_views(views, random_state=self.rs)
        self.Z_ = Z
        V, n = len(Z), len(Z[0])
        self._neighbourhoods(Z)
        joint = np.hstack([z / np.sqrt(z.shape[1]) for z in Z])
        labels = KMeans(self.k, n_init=10, random_state=self.rs).fit_predict(joint)
        w = np.ones(V) / V
        it = 0
        for it in range(1, self.max_iter + 1):
            tp = self._tuple(labels)
            pi = self._pi(tp) if self.use_pi else np.ones(n)
            pi = np.maximum(pi, 1e-6)
            if self.use_w:
                Pv = tp["Pr"] if self.weight_evidence == "raw" else tp["P"]
                r = np.array([(pi * Pv[v]).sum() / pi.sum() for v in range(V)])
                w = r / r.sum()
            D = np.zeros((n, self.k))
            for v, z in enumerate(Z):
                cent = np.stack([(pi[labels == c, None] * z[labels == c]).sum(0) / pi[labels == c].sum()
                                 if (labels == c).any() else z[np.random.default_rng(self.rs + c).integers(n)]
                                 for c in range(self.k)])
                d = ((z[:, None, :] - cent[None]) ** 2).sum(2)
                scale = d[np.arange(n), labels].mean()
                D += w[v] * d / max(scale, 1e-12)
            new = D.argmin(1)
            if np.array_equal(new, labels):
                break
            labels = new
        tp = self._tuple(labels)
        pi = self._pi(tp)
        self.result_ = MVCResult(labels, tp["runner"], w, pi, tp["T"], tp["F"], tp["C"], tp["U"], tp["N"],
                                 tp["G"], it)
        return self.result_


def type_points(res: MVCResult, theta: float = 3.0):
    """Unsupervised typing: each of C, U, N is standardised by its median and IQR over all points; a point takes
    the type of its largest standardised component if it exceeds theta, otherwise 'clean'."""
    S = np.stack([res.C, res.U, res.N], 1)
    med = np.median(S, 0)
    iqr = np.subtract(*np.percentile(S, [75, 25], 0))
    z = (S - med) / np.where(iqr > 1e-12, iqr, 1.0)
    lab = np.array(["conflict", "ambiguous", "outlier"])[z.argmax(1)]
    return np.where(z.max(1) > theta, lab, "clean")
