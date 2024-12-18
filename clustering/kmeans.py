#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import override
from sklearn.cluster import KMeans
from .clustering_base import ClusteringBase

class KMeansCluster(ClusteringBase):

    def __init__(self, num_classes, label_offset = 0, random_state=None):
        super().__init__(num_classes, label_offset)

        self.random_state = random_state
        self.model = KMeans(n_clusters=self.num_classes, random_state=self.random_state)

    @override
    def fit(self, features):
        self.model.fit(features)

    @override
    def _pre_predict(self, features):
        return self.model.predict(features)
