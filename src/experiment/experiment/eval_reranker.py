"""Entry point to run experiments
"""

import sys
import os
import logging
import numpy as np

import tqdm

def nop(it, *a, **k):
    return it
tqdm.tqdm = nop


from omegaconf import DictConfig, OmegaConf
import hydra
from transformers import AutoModel, AutoTokenizer, AutoConfig
from experiment.models.reranker import RerankerDataset, T2TDataCollator
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sentence_transformers import SentenceTransformer
import dataclasses
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from experiment.candidates import get_new_candidates
from experiment.metrics import calc_metrics
from experiment import utils
from search_eval.datasets import base
from experiment.encoders.sentence_transformer import SentTrans
from experiment.index import ann


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()

def clean_pref(metrics:List) -> List:
    prefix = "eval_"
    return {key[len(prefix):]:val for key, val in metrics.items() if key.startswith(prefix)}

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

class SaveReportCallback(TrainerCallback):
    def __init__(self, output_file, base_model, name="reranker"):
        self.output_file = output_file
        self.base_model = base_model
        self.name = name

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        data = {
            "name": self.name,
            "epoch": state.epoch,
            "steps": state.global_step,
            "metrics": clean_pref(metrics),
            "base_model": self.base_model
        }
        self.save_report(data)

    def save_report(self, data):
        utils.save_report(self.output_file, data)
        

def train_reranker(model, tokenizer,
                        ds_train, 
                        ds_candidates, 
                        queries_corpus, 
                        docs_corpus, 
                        compute_metrics,
                        callbacks,
                        cfg):
    train_dataset = RerankerDataset([dict(query=queries_corpus[query], doc=docs_corpus[doc], label=relevancy) for query, doc, relevancy in ds_train.iterrows()])
    test_dataset = RerankerDataset([dict(query=queries_corpus[query], doc=docs_corpus[doc], label=relevancy) for query, doc, relevancy in ds_candidates.iterrows()])
    logger.info(f"Reranker test dataset {len(test_dataset)} items, train dataset {len(train_dataset)} items")
    
    training_args = TrainingArguments(
        output_dir='results',          # output directory
        num_train_epochs=cfg.num_epochs,              # total # of training epochs
        per_device_train_batch_size=cfg.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=cfg.train_batch_size,   # batch size for evaluation
        warmup_steps=100,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='logs',            # directory for storing logs
        evaluation_strategy='steps',
        eval_steps=cfg.eval_steps
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        data_collator=T2TDataCollator(tokenizer, cfg),
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,            # evaluation dataset
        compute_metrics=compute_metrics,
        callbacks=callbacks
        
    )
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
    encoder = SentTrans(cfg.models.senttrans)
    logger.info(f"Encode docs with {encoder.name}")
    encoded_docs = encoder.encode(docs_corpus[ds_test.docs])
    logger.info("Init HNSW index")
    index = ann.HNSW(cfg.index.ann)
    logger.info(f"Indexing docs with {index.name}")
    index.build(encoded_docs)
    logger.info(f"Encode queries with {encoder.name}")
    encoded_queries = encoder.encode(queries_corpus[ds_test.queries_uniq])
    logger.info("Generate candidates for evaluation...")
    ds_candidates = get_new_candidates(index, ds_test, encoded_queries, cfg.reranker.k)
    logger.info(f"Got {len(ds_candidates.docs)} candidate pairs")

    logger.info("Init reranker model")
    tokenizer = AutoTokenizer.from_pretrained(cfg.reranker.base_model)
    reranker = RobertaForSequenceClassification.from_pretrained(cfg.reranker.base_model, num_labels=1)

    _metrics = MyRerankerMetrics(ds_test, ds_candidates, cfg, '@@@')
    _savecallback = SaveReportCallback(cfg.report.output_file, cfg.reranker.base_model)
    #Reranker model
    trainer = train_reranker(reranker, 
                            tokenizer, 
                            ds_train.sample(cfg.reranker.train_sample), 
                            ds_candidates, 
                            queries_corpus, 
                            docs_corpus,
                            _metrics, 
                            [_savecallback],
                            cfg.reranker)
    
    logger.info("Evaluate candidates quality, before training")
    #Evaluate candidates quality
    data = {"name": "Reranker@pre-train", 
            "metrics": calc_metrics(ds_test, ds_candidates), 
            "base_model":cfg.reranker.base_model}
    utils.save_report(cfg.report.output_file, data)
    #Do actual training
    trainer.train()
    #trainer.save_model(cfg.model_save_path)
    data = {"name": "Reranker@post-train", 
            "metrics": clean_pref(trainer.evaluate()),
            "base_model":cfg.reranker.base_model}
    utils.save_report(cfg.report.output_file, data)
    logger.info('All done!')


    

def entry():
    main()

if __name__ == "__main__":
    main()