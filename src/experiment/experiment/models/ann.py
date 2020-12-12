"""Module for managing inverted index
"""

import hnswlib
import logging
import numpy as np


class HNSWIndex:
    def __init__(self, cfg):
        self.cfg = cfg
        self.index = hnswlib.Index(space = self.cfg.space, dim = self.cfg.dim)
    
    def build(self, docs):
        docs = np.asarray(docs)
        self.index.init_index(max_elements = len(docs), ef_construction = self.cfg.ef_construction, M = self.cfg.M)
        self.index.add_items(docs, list(range(len(docs))))
        # Controlling the recall by setting ef:
        #self.index.set_ef(200)  # ef should always be > top_k_hits

    def generate_candidates(self, queries, k=10):
        #queries_embeddings = self.embedder.encode(queries.tolist(), show_progress_bar=True, convert_to_numpy=True)
        corpus_ids, dists = self.index.knn_query(queries, k=k)
        for ids, distances in zip(corpus_ids, dists):
            yield distances, ids