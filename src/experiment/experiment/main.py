"""Entry point to run experiments
"""

import sys
import os
import importlib
from pathlib import Path
import numpy as np
from search_eval.datasets import base
from experiment.models import ann, bm25
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import SentenceEvaluator
from tqdm.auto import tqdm
import torch
from experiment import homedepot
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from collections import defaultdict
from functools import partial
from omegaconf import DictConfig, OmegaConf
import hydra
import random
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from experiment.models.reranker import RerankerDataset, T2TDataCollator
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
import dataclasses
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from experiment.candidates import get_new_candidates
from experiment.metrics import calc_metrics

from transformers.trainer_callback import TrainerCallback

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()

def set_seed(seed):
    logger.info(f"Set seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset(cfg):
    orig_dir = Path(hydra.utils.get_original_cwd())
    data_path = orig_dir / Path(cfg.dataset.dir_path)
    cache_path = orig_dir / Path(cfg.dataset.cache_path)
    docs_corpus_path = orig_dir / Path(cfg.dataset.docs_corpus_path)
    queries_corpus_path = orig_dir / Path(cfg.dataset.queries_corpus_path)
    docs_corpus = None
    queries_corpus = None
    if cache_path.is_file():
        logger.info(f"Loading cache {cache_path}")
        hd = base.Dataset.load_from_cache(cache_path)
        docs_corpus = np.load(docs_corpus_path, allow_pickle=True)
        queries_corpus = np.load(queries_corpus_path, allow_pickle=True)
    else:
        logger.info(f"Cache {cache_path} not found build cache from raw data first.")
        queries_corpus, docs_corpus, relevance = homedepot.import_from_disk(data_path)
        docs_idx = np.arange(docs_corpus.shape[0])
        np.save(docs_corpus_path, docs_corpus, allow_pickle=True)
        queries_corpus, queries_idx = np.unique(queries_corpus, return_inverse=True)
        np.save(queries_corpus_path, queries_corpus, allow_pickle=True)
        hd = base.Dataset(queries_idx, docs_idx, relevance)
        hd.save_to_cache(cache_path)
    return hd, docs_corpus, queries_corpus

class MyRerankerMetrics:
    def __init__(self, ds_test:base.Dataset, ds_candidates:base.Dataset, cfg, name:str = 'reranker'):
        self.ds_test = ds_test
        self.ds_candidates = ds_candidates


    def __call__(self, pred):
        label_ids = pred.label_ids
        preds = pred.predictions.reshape(-1)
        np.testing.assert_equal(label_ids, self.ds_candidates.relevance)
        metrics = calc_metrics(self.ds_test, base.Dataset(self.ds_candidates.queries, self.ds_candidates.docs, preds))
        return metrics


class MyInformationRetrievalEvaluator(SentenceEvaluator):
    """Class for evaluation during experiment
    """
    def __init__(self, ds_test:base.Dataset, queries_corpus, docs_corpus, cfg, name:str = 'sentence_transformer'):
        self.ds_test = ds_test
        self.vectorizer = None
        self.queries_corpus = queries_corpus
        self.docs_corpus = docs_corpus
        self.reranker = None
        self.cfg = cfg
        self.name = name

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info(f"Information Retrieval Evaluation on dataset {out_txt}")
        
        ds_candidates = get_new_candidates(self.ds_test, self.queries_corpus, self.docs_corpus, model, self.cfg, self.cfg.models.senttrans.k)
        metrics = calc_metrics(self.ds_test, ds_candidates)
        data = {
            "name": self.name,
            "epoch": epoch,
            "steps": steps,
            "metrics": metrics,
            "base_model": self.cfg.models.senttrans.base_model
        }
        self.save_report(data)
        max_k = max(self.cfg.metrics.k)
        return metrics[f"ndcg@{max_k}"]

    def save_report(self, data):
        with open(self.cfg.report.output_file, 'a') as f:
            f.write("{}\n".format(json.dumps(data)))

def train_sentencetrans(model, 
                        evaluator, 
                        ds_train, 
                        ds_test, 
                        queries_corpus, 
                        docs_corpus, 
                        cfg):
    train_samples = [InputExample(texts=[queries_corpus[query], docs_corpus[doc]], label=relevancy) for query, doc, relevancy in ds_train.iterrows()]
    #dev_samples = [InputExample(texts=[queries_corpus[query], docs_corpus[doc]], label=relevancy) for query, doc, relevancy in ds_test.iterrows()]
    train_dataset = SentencesDataset(train_samples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)
    # Development set: Measure correlation between cosine score and gold labels
    ###evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='home-depot-dev')
    
    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataset) * cfg.num_epochs / cfg.train_batch_size * 0.05) #5% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    model.fit(train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=cfg.num_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=cfg.model_save_path)
    return model




def train_reranker(model, tokenizer,
                        ds_train, 
                        ds_candidates, 
                        queries_corpus, 
                        docs_corpus, 
                        compute_metrics,
                        cfg):
    train_dataset = RerankerDataset([dict(query=queries_corpus[query], doc=docs_corpus[doc], label=relevancy) for query, doc, relevancy in ds_train.iterrows()])
    test_dataset = RerankerDataset([dict(query=queries_corpus[query], doc=docs_corpus[doc], label=relevancy) for query, doc, relevancy in ds_candidates.iterrows()])
    logger.info(f"Reranker test dataset {len(test_dataset)} items, train dataset {len(train_dataset)} items")
    
    training_args = TrainingArguments(
        output_dir='results',          # output directory
        num_train_epochs=3,              # total # of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=100,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='logs',            # directory for storing logs
        evaluation_strategy='steps',
        eval_steps=500
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        data_collator=T2TDataCollator(tokenizer, cfg),
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,            # evaluation dataset
        compute_metrics=compute_metrics,
        
    )
    trainer.train()
    logger.info(trainer.evaluate())
    #trainer.save_model(cfg.model_save_path)
    return trainer


@hydra.main(config_name="config.yaml")
def main(cfg: DictConfig):
    #Fixing seed
    set_seed(cfg.process.seed)
    #init vars
    #Load dataset
    logger.info("Loading dataset...")
    dataset, docs_corpus, queries_corpus = load_dataset(cfg)
    dataset = dataset.sample(cfg.dataset.sample)
    ds_test, ds_train = dataset.split_train_test(cfg.dataset.test_size)
    ds_test = ds_test#[:200]
    logger.info(f"Train size {len(ds_train)}, Test size {len(ds_test)}")
    logger.info("Init vectorizer model")
    vectorizer = SentenceTransformer(cfg.models.senttrans.base_model)
    logger.info("Init reranker model")
    tokenizer = AutoTokenizer.from_pretrained(cfg.reranker.base_model)
    reranker = RobertaForSequenceClassification.from_pretrained(cfg.reranker.base_model, num_labels=1)

    #Step 2. Reranking
    ds_candidates = get_new_candidates(ds_test, queries_corpus, docs_corpus, vectorizer, cfg, cfg.reranker.k)
    
    _metrics = MyRerankerMetrics(ds_test, ds_candidates, cfg, '@@@')
    #Reranker model
    trainer = train_reranker(reranker, 
                            tokenizer, 
                            ds_train.sample(cfg.reranker.train_sample), 
                            ds_candidates, 
                            queries_corpus, 
                            docs_corpus,
                            _metrics, 
                            cfg.reranker)

    exit(0)

    evaluator = MyInformationRetrievalEvaluator(ds_test, queries_corpus, docs_corpus, cfg)
    #Create vectorizer object
    logger.info(f"Start vectorizer training...")
    train_sentencetrans(vectorizer,
                        evaluator,
                        ds_train.sample(cfg.models.senttrans.train_sample), 
                        ds_test, 
                        queries_corpus, 
                        docs_corpus, 
                        cfg.models.senttrans)
    

    #logger.info(f"Start reranking...")
    #candidates_dataset = RerankerDataset([dict(query=queries_corpus[query], doc=docs_corpus[doc_id]) for query, doc_id, _ in ds_candidates.iterrows()])
    #y = trainer.predict(candidates_dataset).predictions
    #y =list(map(lambda x: x[0], y))

    logger.info('All done!')


    

def entry():
    main()

if __name__ == "__main__":
    main()