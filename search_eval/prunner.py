"""Module for run queries with search models
    algo:
    for each query
        get candidates from model
        calc metrics using matched docs and docs from judgements list
"""
from pathlib import Path
from search_eval.datasets import homedepot
from search_eval.models import bm25
from search_eval.metrics import mapk
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm import tqdm
from search_eval.progressparallel import ProgressParallel

path = "/home/algis/repos/personal/100nuts/semantic-search-domain-adaptation/data/home_depot"
cache_path = "hd_cache.npz"
#hd = homedepot.HomeDepotDataset.import_from_disk(Path(path))
hd = homedepot.HomeDepotDataset.load_from_cache(Path(cache_path))
#hd.save_to_cache(Path(cache_path))
#print(hd)

ds_test, ds_train = hd.split_train_test(0.005)
#print("Test", ds_test)
#print("train", ds_train)
print("--------------------------------")
#exit(0)
print(len(ds_test), len(ds_train))
model = bm25.BM25()
model.build_index(ds_test.docs.tolist())

def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]

def process_chunk(data):
    print(data)
    docs_idx_actual = []
    docs_idx_predicted = []
    rels_actual = []
    rels_predicted = []
    for query, docs, rels in data:
        docs_idx_actual.append(docs)
        rels_actual.append(rels)
        scores, idx = model.generate_candidates(query, 10)
        docs_idx_predicted.append(idx)
        rels_predicted.append(scores)
    return mapk.mapk(docs_idx_actual, docs_idx_predicted, 10)

def preprocess_parallel(data, chunksize=10):
    executor = ProgressParallel(n_jobs=7, backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk)
    tasks = (do(chunk) for chunk in chunker(data, len(data), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)

print(ds_test[0:4])
#print(next(chunker(ds_test, len(ds_test), 3)))
#print(preprocess_parallel(ds_test, 10))