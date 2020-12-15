import sys
from pathlib import Path
import numpy as np
import logging
import math
import json
from experiment import utils
from search_eval.datasets import base
from experiment.models import ann
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, util, InputExample
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from experiment.candidates import get_new_candidates
from experiment.metrics import calc_metrics
from omegaconf import DictConfig, OmegaConf
import hydra
from torch.utils.data import DataLoader

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()

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
        evaluation_steps=cfg.evaluation_steps,
        warmup_steps=warmup_steps,
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
    logger.info("Init vectorizer model")
    vectorizer = SentenceTransformer(cfg.models.senttrans.base_model)

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
    
    logger.info('All done!')


    

def entry():
    main()

if __name__ == "__main__":
    main()