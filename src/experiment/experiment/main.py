"""Entry point to run experiments
"""

import sys
import os
from pathlib import Path
import numpy as np
from search_eval.datasets import base
from experiment.models import bm25
from search_eval.metrics import mapk
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
        self.dataset, self.corpus = self.load_dataset()
        self.ds_test, self.ds_train = self.dataset.split_train_test(cfg.dataset.test_size)
        self.model = bm25.BM25()
        self.model.build_index(self.corpus[self.ds_test.docs].tolist())

    def load_dataset(self):
        orig_dir = Path(hydra.utils.get_original_cwd())
        data_path = orig_dir / Path(self.cfg.dataset.dir_path)
        cache_path = orig_dir / Path(self.cfg.dataset.cache_path)
        corpus_path = orig_dir / Path(self.cfg.dataset.corpus_path)
        hd_corpus = None
        if cache_path.is_file():
            logger.info(f"Loading cache {cache_path}")
            hd = base.Dataset.load_from_cache(cache_path)
            hd_corpus = np.load(corpus_path, allow_pickle=True)
        else:
            logger.info(f"Cache {cache_path} not found build cache from raw data first.")
            queries, hd_corpus, relevance = homedepot.import_from_disk(data_path)
            docs_idx = np.arange(hd_corpus.shape[0])
            np.save(corpus_path, hd_corpus, allow_pickle=True)
            hd = base.Dataset(queries, docs_idx, relevance)
            hd.save_to_cache(cache_path)
        return hd, hd_corpus

    def process(self, data):
        res = []
        for query, docs, rels in data:
            scores, idx = self.model.generate_candidates(query, 10)
            res.append(mapk.apk(docs, self.ds_test.docs[idx], 10))
        return res

def container_fun(experiment_obj, data):
    return experiment_obj.process(data)


@hydra.main(config_name="config.yaml")
def main(cfg: DictConfig):
    
    logger.info(OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.process.seed)
    e = Experiment(cfg)
    res = process_parallel(partial(container_fun, e), e.ds_test, cfg.process.batch)
    #logger.info(res)
    logger.info(np.mean(np.array(flatten(res))))

def entry():
    main()

if __name__ == "__main__":
    main()