"""Model for search docs using Sentence-Transformers
"""

from sentence_transformers import SentenceTransformer, util
import scipy.spatial
import logging
import numpy as np
import torch
from search_eval.models import base
from search_eval.datasets.base import Dataset as SearchEvalDS
import hnswlib
from typing import List

from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

logger = logging.getLogger(__name__)

def sent_dataset(dataset: SearchEvalDS, queries_corpus, docs_corpus) -> List[InputExample]:
    queries_corpus = np.asarray(queries_corpus)
    docs_corpus = np.asarray(docs_corpus)
    examples = []
    for query, docs, rels in dataset:
        query_text = queries_corpus[query]
        for doc, relevancy in zip(docs, rels):
            doc_text = docs_corpus[doc]
            examples.append(InputExample(texts=[query_text, doc_text], label=relevancy))
    return examples

train_batch_size=96
num_epochs=2

class SentTrans(base.Model):
    def __init__(self, model_path='distilroberta-base-msmarco-v2', embedding_size=768):
        self.model_path = model_path
        self.embedding_size = embedding_size
        self.index = None
        self.embedder = None

    def init_embedder(self):
        self.embedder = SentenceTransformer(self.model_path)

    def build_index(self, docs: List[str]):
        docs = list(docs)
        self.index = hnswlib.Index(space = 'cosine', dim = self.embedding_size)
        if not self.embedder:
            self.init_embedder()
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
    
    def train(self, train_samples, dev_samples, model_save_path):
        if not self.embedder:
            self.init_embedder()
        train_dataset = SentencesDataset(train_samples, self.embedder)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.CosineSimilarityLoss(model=self.embedder)
        # Development set: Measure correlation between cosine score and gold labels
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='home-depot-dev')
        # Configure the training. We skip evaluation in this example
        warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        self.embedder.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)