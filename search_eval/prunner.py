"""Module for run queries with search models
    algo:
    for each query
        get candidates from model
        calc metrics using matched docs and docs from judgements list
"""
import sys
from pathlib import Path
import numpy as np
from search_eval.datasets import base
from search_eval.models import bm25
from search_eval.metrics import mapk
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm import tqdm
from search_eval.progressparallel import ProgressParallel
from search_eval import homedepot
import logging


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()

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


np.random.seed(42)
logger.info(f"Split dataset into test/train.") 
ds_test, ds_train = hd.split_train_test(0.005)

logger.debug(f"Test {ds_test} \n\nTrain {ds_train}") 


model = bm25.BM25()
model.build_index(hd_corpus[ds_test.docs].tolist())


def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]

def process_chunk(data):
    docs_idx_actual = []
    docs_idx_predicted = []
    rels_actual = []
    rels_predicted = []
    res = []
    for query, docs, rels in data:
        docs_idx_actual.append(docs)
        rels_actual.append(rels)
        scores, idx = model.generate_candidates(query, 10)
        docs_idx_predicted.append(ds_test.docs[idx])
        rels_predicted.append(scores)
        res.append(mapk.apk(docs, ds_test.docs[idx], 10))
    return res

def preprocess_parallel(data, chunksize=10):
    executor = ProgressParallel(n_jobs=7, backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk)
    tasks = (do(chunk) for chunk in chunker(data, len(data), chunksize=chunksize))
    result = executor(tasks)
    #return result
    return flatten(result)


#print(process_chunk(ds_test))
res = preprocess_parallel(ds_test, 100)
print(np.mean(np.array(res)))