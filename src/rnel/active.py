"""Label-efficient multi-view active learning with RNEL reliability.

The pipeline has two RNEL components:
  cold start   RNEL-MVC is fitted once on the unlabelled pool; the first labels are, in each cluster, the object
               with the highest point weight pi (diverse and reliable);
  queries      the classifier margin 1 - (p1 - p2) is multiplied by the RNEL-MVC reliability
               min(1, pi / median(pi)), so that objects whose views conflict (likely corrupted) are rarely asked.
Preprocessing (standardisation and PCA per view) is fitted on the pool only. The budget is exact: the last batch
is truncated so that exactly `budget` labels are requested.

Example
-------
>>> learner = ReliabilityActiveLearner(n_classes=5, budget=100)
>>> learner.fit_pool(views_pool)                 # list of (n_pool, p_v) arrays
>>> for idx in learner.queries(oracle):          # oracle(indices) -> labels
...     pass
>>> learner.predict(views_test)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from .mvc import RNELMVC, ViewEmbedder


def reliability(point_weights: np.ndarray) -> np.ndarray:
    """RNEL-MVC reliability in [0, 1]: pi / median(pi), capped at 1."""
    med = np.median(point_weights)
    return np.minimum(1.0, point_weights / med) if med > 0 else np.ones_like(point_weights)


def margin(proba: np.ndarray) -> np.ndarray:
    s = np.sort(proba, 1)
    return 1 - (s[:, -1] - s[:, -2]) if proba.shape[1] > 1 else np.zeros(len(proba))


class ReliabilityActiveLearner:
    def __init__(self, n_classes: int, budget: int = 200, batch: int = 10, cold_start: str = "rnel",
                 query: str = "margin_reliability", random_state: int = 0, **mvc_kwargs):
        if cold_start not in ("rnel", "random", "kmeans"):
            raise ValueError("cold_start must be 'rnel', 'random' or 'kmeans'")
        if query not in ("margin_reliability", "margin", "random"):
            raise ValueError("query must be 'margin_reliability', 'margin' or 'random'")
        self.k, self.budget, self.batch = n_classes, budget, batch
        self.cold, self.query, self.rs, self.mvc_kwargs = cold_start, query, random_state, mvc_kwargs

    def fit_pool(self, views_pool: list[np.ndarray]) -> "ReliabilityActiveLearner":
        self.emb_ = ViewEmbedder(random_state=self.rs).fit(views_pool)
        Z = self.emb_.transform(views_pool)
        self.J_ = np.hstack([z / np.sqrt(z.shape[1]) for z in Z])
        self.n_ = len(self.J_)
        self.mvc_ = RNELMVC(self.k, random_state=self.rs, **self.mvc_kwargs).fit(views_pool)
        self.rel_ = reliability(self.mvc_.point_weights)
        self.rng_ = np.random.default_rng(self.rs)
        return self

    def _initial(self) -> list[int]:
        if self.cold == "random":
            return list(self.rng_.choice(self.n_, min(self.k, self.budget), replace=False))
        labels = self.mvc_.labels if self.cold == "rnel" else KMeans(self.k, n_init=10, random_state=self.rs).fit_predict(self.J_)
        out = []
        for c in range(self.k):
            m = np.where(labels == c)[0]
            if len(m) == 0:
                continue
            if self.cold == "rnel":
                out.append(int(m[np.argmax(self.mvc_.point_weights[m])]))
            else:  # k-means: object closest to the cluster mean
                out.append(int(m[np.argmin(((self.J_[m] - self.J_[m].mean(0)) ** 2).sum(1))]))
        return out[: self.budget]

    def _fit_classifier(self):
        yL = np.array(self.y_)
        if len(np.unique(yL)) > 1:
            self.clf_ = LogisticRegression(max_iter=2000).fit(self.J_[self.L_], yL)
        else:
            self.clf_ = None

    def queries(self, oracle: Callable[[np.ndarray], np.ndarray]):
        """Generator: yields each batch of pool indices after it has been labelled by `oracle`."""
        self.L_ = self._initial()
        self.y_ = list(oracle(np.array(self.L_)))
        self._fit_classifier()
        yield np.array(self.L_)
        while len(self.L_) < self.budget:
            b = min(self.batch, self.budget - len(self.L_))
            U = np.setdiff1d(np.arange(self.n_), self.L_)
            if self.query == "random" or self.clf_ is None:
                q = self.rng_.choice(U, b, replace=False)
            else:
                score = margin(self.clf_.predict_proba(self.J_[U]))
                if self.query == "margin_reliability":
                    score = score * self.rel_[U]
                q = U[np.argsort(-score + 1e-9 * self.rng_.random(len(U)))[:b]]
            self.L_.extend(int(i) for i in q)
            self.y_.extend(oracle(q))
            self._fit_classifier()
            yield q

    def predict(self, views: list[np.ndarray]) -> np.ndarray:
        Z = self.emb_.transform(views)
        J = np.hstack([z / np.sqrt(z.shape[1]) for z in Z])
        if self.clf_ is None:
            return np.full(len(J), self.y_[0])
        return self.clf_.predict(J)
