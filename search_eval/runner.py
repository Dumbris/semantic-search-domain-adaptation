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
from tqdm import tqdm

path = "/home/algis/repos/personal/100nuts/semantic-search-domain-adaptation/data/home_depot"
cache_path = "hd_cache.npz"
#hd = homedepot.HomeDepotDataset.import_from_disk(Path(path))
hd = homedepot.HomeDepotDataset.load_from_cache(Path(cache_path))
#hd.save_to_cache(Path(cache_path))
#print(hd)

ds_test, ds_train = hd.split_train_test(0.1)
#print("Test", ds_test)
#print("train", ds_train)
print("--------------------------------")
#exit(0)
print(len(ds_test), len(ds_train))
model = bm25.BM25()
model.build_index(ds_test.docs.tolist())
docs_idx_actual = []
docs_idx_predicted = []
rels_actual = []
rels_predicted = []
for query, docs, rels in tqdm(ds_test):
    docs_idx_actual.append(docs)
    rels_actual.append(rels)
    scores, idx = model.generate_candidates(query, 10)
    docs_idx_predicted.append(idx)
    rels_predicted.append(scores)

print(mapk.mapk(docs_idx_actual, docs_idx_predicted, 10))