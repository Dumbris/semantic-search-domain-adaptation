import sys
from pathlib import Path
import numpy as np
import logging
import math
import json
from experiment import utils
from search_eval.datasets import base
from experiment.index import bm25
from experiment.encoders.tokenizer import Tokenizer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from experiment.candidates import get_new_candidates
from experiment.metrics import calc_metrics
from omegaconf import DictConfig, OmegaConf
import hydra


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()




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
    encoder = Tokenizer(cfg.encoders.tokenizer)
    logger.info(f"Encode docs with {encoder.name}")
    encoded_docs = encoder.encode(docs_corpus[ds_test.docs])
    logger.info("Init bm25 model")
    index = bm25.BM25()
    logger.info(f"Indexing docs with {index.name}")
    index.build(encoded_docs)
    logger.info(f"Encode queries with {encoder.name}")
    encoded_queries = encoder.encode(queries_corpus[ds_test.queries_uniq])
    logger.info("Generate candidates for evaluation...")
    ds_candidates = get_new_candidates(index, ds_test, encoded_queries, cfg.index.bm25.k)
    metrics = calc_metrics(ds_test, ds_candidates)
    data = {
        "name": "BM25",
        "metrics": metrics,
        "base_model": ""
    }
    utils.save_report(cfg.report.output_file, data)
    
    logger.info('All done!')


    

def entry():
    main()

if __name__ == "__main__":
    main()