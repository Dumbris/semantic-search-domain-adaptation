from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
from search_eval.models import base
from typing import List
from experiment.models.preprocessing import Preprocess


class BM25(base.Model):
    def __init__(self):
        self.preproc = Preprocess()
        self.corpus = None
        self.index = None

    def build_index(self, docs: List[str]):
        docs = list(docs)
        self.corpus = self.preproc.preprocess_pipe(docs)
        #self.corpus = [' '.join(x) for x in tokenized_docs_list]
        self.index = BM25Okapi(self.corpus)

    def generate_candidates(self, query, k=10):
        tokenized_query = self.preproc.preprocess_pipe([query])[0]
        scores = self.index.get_scores(tokenized_query)
        idx = np.argsort(scores)[::-1][:k]
        #idx = np.argpartition(scores, -1*k)[-1*k:]
        return scores[idx], idx