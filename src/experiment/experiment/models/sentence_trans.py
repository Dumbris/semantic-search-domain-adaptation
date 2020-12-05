"""Model for search docs using Sentence-Transformers
"""

from sentence_transformers import SentenceTransformer, util
import scipy.spatial
import logging
import numpy as np
import torch
from search_eval.models import base
from typing import List


class SentTrans(base.Model):
    def __init__(self, model_path='distilroberta-base-msmarco-v2'):
        self.index = None #Corpus embeddings
        self.embedder = SentenceTransformer(model_path)

    def build_index(self, docs: List[str]):
        docs = list(docs)
        self.index = self.embedder.encode(docs, convert_to_tensor=True)

    def generate_candidates(self, query, k=10):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.index)[0]
        idx = torch.topk(scores,k).indices.numpy()
        return scores[idx].numpy(), idx