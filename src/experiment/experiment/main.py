"""Entry point to run experiments
"""

import sys
import os
from pathlib import Path
import numpy as np
from search_eval.datasets import base
from experiment.models import bm25, sentence_trans
from search_eval.metrics import mapk, ndcg
from search_eval.models import oracle
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm import tqdm
from search_eval.progressparallel import process_parallel, flatten
from experiment import homedepot
import logging
from collections import defaultdict
from functools import partial
from omegaconf import DictConfig, OmegaConf
import hydra


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()

TRANSLATE_IDS = True
USE_PARALLEL  = False

class Experiment:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.dataset, self.docs_corpus, self.queries_corpus = self.load_dataset()
        self.ds_test, self.ds_train = self.dataset.split_train_test(cfg.dataset.test_size)
        self.model = model
        self.model.build_index(self.docs_corpus[self.ds_test.docs].tolist())

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
        res = defaultdict(list)
        for query, docs, rels in data:
            if TRANSLATE_IDS:
                #for bm25 model we need to translate ids
                scores, idx = self.model.generate_candidates(self.queries_corpus[query], 10)
                ids_pred = self.ds_test.docs[idx]
            else:
                scores, idx = self.model.generate_candidates(query, 10)
                ids_pred = idx
            for k in [3,5,10]:
                res[f"apk@{k}"].append(mapk.apk(docs, ids_pred, k))
                res[f"ndcg@{k}"].append(ndcg.ndcg(rels, docs, scores, ids_pred, k))
        return res

def container_fun(experiment_obj, data):
    return experiment_obj.process(data)

def merge_metrics(arr):
    res = defaultdict(list)
    for run_result in arr:
        for k, v in run_result.items():
            res[k].extend(v)
    return res

@hydra.main(config_name="config.yaml")
def main(cfg: DictConfig):
    np.random.seed(cfg.process.seed)
    model = sentence_trans.SentTrans('distilroberta-base-msmarco-v2')
    #model = bm25.BM25()
    #model = oracle.Oracle()
    #model.build_index(self.ds_test)
    e = Experiment(cfg, model)
    logger.info(f"Test size {len(e.ds_test)}")
    if USE_PARALLEL:
        res = process_parallel(partial(container_fun, e), e.ds_test, cfg.process.batch)
        metrics = merge_metrics(res)
    else:
        metrics = e.process(e.ds_test)
    for name, vals in metrics.items():
        val = np.mean(np.array(vals))
        logger.info(f"{name}: {val}")

def entry():
    main()

if __name__ == "__main__":
    main()