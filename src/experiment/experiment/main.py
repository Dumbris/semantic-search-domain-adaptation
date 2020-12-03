"""Entry point to run experiments
"""

import sys
from pathlib import Path
import numpy as np
from search_eval.datasets import base
from experiment.models import bm25
from search_eval.metrics import mapk
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm import tqdm
from search_eval.progressparallel import process_parallel
from experiment import homedepot
import logging
from functools import partial


np.random.seed(42)

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()

class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_dataset(self):
        data_path = Path("./data/home_depot")
        cache_path = data_path / Path("hd_cache.npz")
        corpus_path = data_path / Path("hd_corpus.npy")
        hd_corpus = None
        if cache_path.is_file():
            logger.info(f"Loading cache {cache_path}")
            hd = base.Dataset.load_from_cache(cache_path)
            hd_corpus = np.load(corpus_path, allow_pickle=True)
        else:
            logger.info(f"Cache {cache_path} not found build cache from raw data first.")
            queries, hd_corpus, relevance = homedepot.import_from_disk(Path(data_path))
            docs_idx = np.arange(hd_corpus.shape[0])
            np.save(corpus_path, hd_corpus, allow_pickle=True)
            hd = base.Dataset(queries, docs_idx, relevance)
            hd.save_to_cache(cache_path)
        return hd, hd_corpus

    def process(self, data):
        res = []
        for query, docs, rels in data:
            scores, idx = self.model.generate_candidates(query, 10)
            res.append(mapk.apk(docs, ds_test.docs[idx], 10))
        return res

def container_fun(experiment_obj, data):
    experiment_obj.run(data)


def main():
    hd, hd_corpus = load_dataset()
    ds_test, ds_train = hd.split_train_test(0.005)
    model = bm25.BM25()
    model.build_index(hd_corpus[ds_test.docs].tolist())
    process_parallel(partial(process, model), ds_test, 400)


if __name__ == "__main__":
    main()