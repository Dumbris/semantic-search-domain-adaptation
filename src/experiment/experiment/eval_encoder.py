import sys
from pathlib import Path
import numpy as np
import logging
import math
import json
import tqdm

def nop(it, *a, **k):
    return it
tqdm.tqdm = nop

from experiment import utils
from search_eval.datasets import base
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, util, InputExample
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from experiment.candidates import get_new_candidates
from experiment.metrics import calc_metrics
from omegaconf import DictConfig, OmegaConf
import hydra
from torch.utils.data import DataLoader
from experiment.encoders.sentence_transformer import SentTrans
from experiment.index import ann

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()

class MyInformationRetrievalEvaluator(SentenceEvaluator):
    """Class for evaluation during experiment
    """
    def __init__(self, 
                    index, 
                    ds_test:base.Dataset, 
                    queries_corpus, 
                    docs_corpus, 
                    report_file, 
                    max_k, 
                    k, 
                    base_model: str,
                    name:str = 'sentence_transformer'):
        self.index = index
        self.ds_test = ds_test
        self.queries_corpus = queries_corpus
        self.docs_corpus = docs_corpus
        self.report_file = report_file
        self.k = k
        self.max_k = max_k
        self.name = name
        self.base_model = base_model #For logging

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info(f"Information Retrieval Evaluation on dataset {out_txt}")
        
        
        encoded_docs = model.encode(self.docs_corpus[self.ds_test.docs])
        logger.info(f"Indexing docs with {self.index.name}")
        self.index.build(encoded_docs)
        encoded_queries = model.encode(self.queries_corpus[self.ds_test.queries_uniq])
        logger.info("Generate candidates for evaluation...")
        ds_candidates = get_new_candidates(self.index, self.ds_test, encoded_queries, self.k)
        logger.info(f"Got {len(ds_candidates.docs)} candidate pairs")
        metrics = calc_metrics(self.ds_test, ds_candidates)
        data = {
            "name": self.name,
            "epoch": epoch,
            "steps": steps,
            "metrics": metrics,
            "base_model": self.base_model
        }
        utils.save_report(self.report_file, data)
        return metrics[f"ndcg@{self.max_k}"]


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
        evaluation_steps=cfg.evaluation_steps,
        warmup_steps=warmup_steps,
        save_best_model=False,
        output_path=cfg.model_save_path)

    return model

@hydra.main(config_name="config_encoder.yaml")
def main(cfg: DictConfig):
    #Fixing seed
    utils.set_seed(cfg.process.seed)
    #init vars
    #Load dataset
    logger.info("Loading dataset...")
    dataset, docs_corpus, queries_corpus = utils.load_dataset(cfg)
    dataset = dataset.sample(cfg.dataset.sample)
    ds_test, ds_train = dataset.split_train_test(cfg.dataset.test_size)
    ds_test = ds_test#[:200]
    logger.info(f"Train size {len(ds_train)}, Test size {len(ds_test)}")
    
    logger.info("Init HNSW index")
    index = ann.HNSW(cfg.index.ann)

    evaluator = MyInformationRetrievalEvaluator(index, 
                                                ds_test, 
                                                queries_corpus, 
                                                docs_corpus,
                                                cfg.report.output_file, 
                                                max(cfg.metrics.k), 
                                                cfg.models.senttrans.k,
                                                cfg.models.senttrans.base_model
                                                )
    #Create vectorizer object

    logger.info("Init vectorizer model")
    encoder = SentTrans(cfg.models.senttrans)
    logger.info(f"Start vectorizer training...")
    train_sentencetrans(encoder.encoder,
                        evaluator,
                        ds_train.sample(cfg.models.senttrans.train_sample), 
                        ds_test, 
                        queries_corpus, 
                        docs_corpus, 
                        cfg.models.senttrans)
    
    logger.info('All done!')


    

def entry():
    main()

if __name__ == "__main__":
    main()