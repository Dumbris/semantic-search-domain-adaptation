"""Model for search docs using Sentence-Transformers
"""

from sentence_transformers import SentenceTransformer, util
import scipy.spatial
import logging
import numpy as np
import torch
from search_eval.models import base
import hnswlib
from typing import List

logger = logging.getLogger(__name__)


class SentTrans(base.Model):
    def __init__(self, model_path='distilroberta-base-msmarco-v2', embedding_size=768):
        self.model_path = model_path
        self.embedding_size = embedding_size
        self.index = None
        self.embedder = None

    def build_index(self, docs: List[str]):
        docs = list(docs)
        self.index = hnswlib.Index(space = 'cosine', dim = self.embedding_size)
        self.embedder = SentenceTransformer(self.model_path)
        logger.info("Encode the corpus. This might take a while")
        corpus_embeddings = self.embedder.encode(docs, show_progress_bar=True, convert_to_numpy=True)
        logger.info("Start creating HNSWLIB index")
        self.index.init_index(max_elements = len(corpus_embeddings), ef_construction = 400, M = 64)
        self.index.add_items(corpus_embeddings, list(range(len(corpus_embeddings))))
        # Controlling the recall by setting ef:
        self.index.set_ef(80)  # ef should always be > top_k_hits

    def generate_candidates(self, queries, k=10):
        queries_embeddings = self.embedder.encode(queries.tolist(), show_progress_bar=True, convert_to_numpy=True)
        corpus_ids, dists = self.index.knn_query(queries_embeddings, k=k)
        for ids, distances in zip(corpus_ids, dists):
            yield distances, ids