"""Module for run queries with search models
    algo:
    for each query
        get candidates from model
        calc metrics using matched docs and docs from judgements list
"""
from pathlib import Path
from search_eval.datasets import homedepot

path = "/home/algis/repos/personal/100nuts/semantic-search-domain-adaptation/data/home_depot"
cache_path = "hd_cache.npz"
#hd = homedepot.HomeDepotDataset.import_from_disk(Path(path))
hd = homedepot.HomeDepotDataset.load_from_cache(Path(cache_path))
#hd.save_to_cache(Path(cache_path))
#print(hd)

ds_test, ds_train = hd.split_train_test(0.05)
print("Test", ds_test)
print("train", ds_train)
print("--------------------------------")
for i in range(5):
    print(ds_test[i])
    print("------")