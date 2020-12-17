from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
from search_eval.models import base
from typing import List
from experiment.models.preprocessing import Preprocess
from search_eval.progressparallel import ProgressParallel, process_parallel, chunker


class BM25(base.Model):
    def __init__(self):
        self.preproc = Preprocess()
        self.index = None

    def build_index(self, docs: List[str]):
        docs = list(docs)
        self.index = BM25Okapi(self.preproc.preprocess_pipe(docs))

    def get_candidates_1q(self, tokenized_query, k=10):
        scores = self.index.get_scores(tokenized_query)
        idx = np.argsort(scores)[::-1][:k]
        return scores[idx], idx

    def process_chunk(self, queries):
        docs_idx_actual = []
        docs_idx_predicted = []
        rels_actual = []
        rels_predicted = []
        res = []
        for query, docs, rels in data:
            docs_idx_actual.append(docs)
            rels_actual.append(rels)
            scores, idx = model.generate_candidates(query, 10)
            docs_idx_predicted.append(ds_test.docs[idx])
            rels_predicted.append(scores)
            res.append(mapk.apk(docs, ds_test.docs[idx], 10))
        return res

    def generate_candidates(self, queries, k=10, chunksize=10):
        tokenized_queries = self.preproc.preprocess_pipe(queries)
        #TODO pass k?
        result = process_parallel(self.process_chunk, tokenized_queries, chunksize)
        return result
        #return flatten(result)