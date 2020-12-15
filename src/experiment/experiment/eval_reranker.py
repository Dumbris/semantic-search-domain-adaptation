"""Entry point to run experiments
"""

import sys
import os
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
from transformers import AutoModel, AutoTokenizer, AutoConfig
from experiment.models.reranker import RerankerDataset, T2TDataCollator
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import dataclasses
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from experiment.candidates import get_new_candidates
from experiment.metrics import calc_metrics
from experiment import utils
from search_eval.datasets import base


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()

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
    
@hydra.main(config_name="config_reranker.yaml")
def main(cfg: DictConfig):
    #Fixing seed
    utils.set_seed(cfg.process.seed)
    #Load dataset
    logger.info("Loading dataset...")
    dataset, docs_corpus, queries_corpus = utils.load_dataset(cfg)
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

    logger.info('All done!')


    

def entry():
    main()

if __name__ == "__main__":
    main()