import random
import torch
import numpy as np

from experiment import homedepot
from pathlib import Path
import numpy as np
import hydra
import logging
from search_eval.datasets import base
import json

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset(cfg):
    logger.info("Loading dataset...")
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
        logger.info(f"Save dataset into cache files {cache_path}")
        np.save(docs_corpus_path, docs_corpus, allow_pickle=True)
        queries_corpus, queries_idx = np.unique(queries_corpus, return_inverse=True)
        np.save(queries_corpus_path, queries_corpus, allow_pickle=True)
        hd = base.Dataset(queries_idx, docs_idx, relevance)
        hd.save_to_cache(cache_path)
    logger.info(f"Loaded {len(hd)} items.")
    return hd, docs_corpus, queries_corpus



def save_report(output_file, data):
    with open(output_file, 'a') as f:
        f.write("{}\n".format(json.dumps(data)))