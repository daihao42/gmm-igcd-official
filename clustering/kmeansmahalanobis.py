#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from typing import override
from .clustering_base import ClusteringBase

class KMeansMahalanobis(KMeans):
    def __init__(self, n_clusters=8, **kwargs):
        super().__init__(n_clusters=n_clusters, **kwargs)

    def fit(self, X, y=None):
        # Compute the covariance matrix and its inverse
        self.cov_matrix_ = np.cov(X, rowvar=False)
        self.inv_cov_matrix_ = np.linalg.inv(self.cov_matrix_)
        
        return super().fit(X, y)

    def _mahalanobis_distances(self, X, centers):
        return cdist(X, centers, metric='mahalanobis', VI=self.inv_cov_matrix_)

    def _assign_labels(self, X, centers):
        # Use Mahalanobis distance for assigning labels
        distances = self._mahalanobis_distances(X, centers)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self._assign_labels(X, self.cluster_centers_)

    def predict(self, X):
        return self._assign_labels(X, self.cluster_centers_)


class KMeansMahalanobisCluster(ClusteringBase):

    def __init__(self, num_classes, label_offset = 0, random_state=None):
        super().__init__(num_classes, label_offset)

        self.random_state = random_state
        self.model = KMeansMahalanobis(n_clusters=self.num_classes, random_state=self.random_state)

    @override
    def fit(self, features):
        self.model.fit(features)

    @override
    def _pre_predict(self, features):
        return self.model.predict(features)
