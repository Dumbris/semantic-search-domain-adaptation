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
from search_eval.metrics import mapk, ndcg
from search_eval.models import oracle
from tqdm.auto import tqdm
import torch
#from joblib import Parallel, delayed
from tqdm import tqdm
#from search_eval.progressparallel import process_parallel, flatten
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
from experiment.models.reranker import RerankerDataset, T2TDataCollator, compute_metrics
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

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

def train_sentencetrans(model, 
                        ds_train, 
                        ds_test, 
                        queries_corpus, 
                        docs_corpus, 
                        cfg):
    train_samples = [InputExample(texts=[queries_corpus[query], docs_corpus[doc]], label=relevancy) for query, doc, relevancy in ds_train.iterrows()]
    dev_samples = [InputExample(texts=[queries_corpus[query], docs_corpus[doc]], label=relevancy) for query, doc, relevancy in ds_test.iterrows()]
    train_dataset = SentencesDataset(train_samples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)
    # Development set: Measure correlation between cosine score and gold labels
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='home-depot-dev')
    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataset) * cfg.num_epochs / cfg.train_batch_size * 0.1) #10% of train data for warm-up
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
                        ds_test, 
                        queries_corpus, 
                        docs_corpus, 
                        cfg):
    train_dataset = RerankerDataset([dict(query=queries_corpus[query], doc=docs_corpus[doc], label=relevancy) for query, doc, relevancy in ds_train.iterrows()])
    test_dataset = RerankerDataset([dict(query=queries_corpus[query], doc=docs_corpus[doc], label=relevancy) for query, doc, relevancy in ds_train.iterrows()])
    
    training_args = TrainingArguments(
        output_dir='results',          # output directory
        num_train_epochs=2,              # total # of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=100,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='logs',            # directory for storing logs
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        data_collator=T2TDataCollator(tokenizer, cfg),
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset            # evaluation dataset
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model(cfg.model_save_path)

@hydra.main(config_name="config.yaml")
def main(cfg: DictConfig):
    #Fixing seed
    set_seed(cfg.process.seed)
    #init vars

    #Load dataset
    dataset, docs_corpus, queries_corpus = load_dataset(cfg)
    ds_test, ds_train = dataset.split_train_test(cfg.dataset.test_size)
    logger.info(f"Train size {len(ds_train)}")
    logger.info(f"Test size {len(ds_test)}")

        #Reranker model
    tokenizer = AutoTokenizer.from_pretrained(cfg.reranker.base_model)
    reranker = RobertaForSequenceClassification.from_pretrained(cfg.reranker.base_model, num_labels=1)
    train_reranker(reranker, tokenizer, ds_train, ds_test, queries_corpus, docs_corpus, cfg.reranker)

    #Create vectorizer object
    vectorizer = SentenceTransformer(cfg.models.senttrans.base_model)
    #logger.info(f"Start training...")
    #vectorizer = train_sentencetrans(vectorizer, ds_train, ds_test, queries_corpus, docs_corpus, cfg.models.senttrans)

    #Init index class
    index = ann.HNSWIndex(cfg.index.ann)
    logger.info(f"Encode docs...")
    vectorized_docs = vectorizer.encode(docs_corpus[ds_test.docs].tolist(), show_progress_bar=True, convert_to_numpy=True)
    logger.info(f"Indexing docs...")
    index.build(vectorized_docs)

    logger.info(f"Encode queries...")
    queries_list = queries_corpus[ds_test.queries_uniq].tolist()
    vectorized_queries = vectorizer.encode(queries_list, show_progress_bar=True, convert_to_numpy=True)
    metrics = defaultdict(list)
    for (scores, idx), item in zip(index.generate_candidates(vectorized_queries, 10), ds_test):
        query, docs, rels = item
        ids_pred = ds_test.docs[idx]
        for k in [3,5,10]:
            metrics[f"apk@{k}"].append(mapk.apk(docs, ids_pred, k))
            metrics[f"ndcg@{k}"].append(ndcg.ndcg(rels, docs, scores, ids_pred, k))

    for name, vals in metrics.items():
        val = np.mean(np.array(vals))
        logger.info(f"{name}: {val}")


    

def entry():
    main()

if __name__ == "__main__":
    main()