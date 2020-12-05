"""Entry point to run experiments
"""

import sys
import os
from pathlib import Path
import numpy as np
from search_eval.datasets import base
from experiment.models import bm25
from search_eval.metrics import mapk, ndcg
from search_eval.models import oracle
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm import tqdm
from search_eval.progressparallel import process_parallel, flatten
from experiment import homedepot
import logging
from functools import partial
from omegaconf import DictConfig, OmegaConf
import hydra


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()

class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset, self.docs_corpus, self.queries_corpus = self.load_dataset()
        self.ds_test, self.ds_train = self.dataset.split_train_test(cfg.dataset.test_size)
        #self.model = bm25.BM25()
        #self.model.build_index(self.docs_corpus[self.ds_test.docs].tolist())
        self.model = oracle.Oracle()
        self.model.build_index(self.ds_test)

    def load_dataset(self):
        orig_dir = Path(hydra.utils.get_original_cwd())
        data_path = orig_dir / Path(self.cfg.dataset.dir_path)
        cache_path = orig_dir / Path(self.cfg.dataset.cache_path)
        docs_corpus_path = orig_dir / Path(self.cfg.dataset.docs_corpus_path)
        queries_corpus_path = orig_dir / Path(self.cfg.dataset.queries_corpus_path)
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

    def process(self, data):
        res = []
        for query, docs, rels in data:
            #scores, idx = self.model.generate_candidates(self.queries_corpus[query], 10)
            scores, idx = self.model.generate_candidates(query, 10)
            ids_pred = self.ds_test.docs[idx]
            _apk = mapk.apk(docs, ids_pred, 10)
            _ndcg = ndcg.ndcg(rels, docs, scores, ids_pred, 10)
            res.append((_apk, _ndcg))
        return res

def container_fun(experiment_obj, data):
    return experiment_obj.process(data)


@hydra.main(config_name="config.yaml")
def main(cfg: DictConfig):
    
    logger.info(OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.process.seed)
    e = Experiment(cfg)
    logger.info(f"Test size {len(e.ds_test)}")
    res = process_parallel(partial(container_fun, e), e.ds_test, cfg.process.batch)
    logger.info(res)
    #logger.info(np.mean(np.array(flatten(res))))

def entry():
    main()

if __name__ == "__main__":
    main()