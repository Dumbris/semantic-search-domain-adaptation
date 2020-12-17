"""Module for managing inverted index
"""

import hnswlib
import logging
import numpy as np
from experiment.index.base import BaseIndex


class HNSW(BaseIndex):
    def __init__(self, cfg, name="HNSW"):
        self.cfg = cfg
        self.name = name
        self.index = None
    
    def build(self, docs):
        docs = np.asarray(docs)
        self.index = hnswlib.Index(space = self.cfg.space, dim = self.cfg.dim)
        self.index.init_index(max_elements = len(docs), ef_construction = self.cfg.ef_construction, M = self.cfg.M)
        self.index.add_items(docs, list(range(len(docs))))

    def generate_candidates(self, queries, k=10):
        corpus_ids, dists = self.index.knn_query(queries, k=k)
        for ids, distances in zip(corpus_ids, dists):
            yield distances, ids