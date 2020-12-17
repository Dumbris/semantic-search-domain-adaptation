from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
from search_eval.models import base
from typing import List
from experiment.index.base import BaseIndex


class BM25(BaseIndex):
    def __init__(self, cfg=None, name="BM25"):
        self.cfg = cfg
        self.name = name
        self.index = None

    def build(self, encoded_docs: List):
        self.index = BM25Okapi(list(encoded_docs))

    def generate_candidates(self, encoded_queries: List, k=10):
        for tokenized_query in encoded_queries:
            scores = self.index.get_scores(tokenized_query)
            idx = np.argsort(scores)[::-1][:k]
            yield scores[idx], idx