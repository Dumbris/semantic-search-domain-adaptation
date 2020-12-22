"""Module contains code for loading Home Depot dataset
"""
import pandas as pd
import numpy as np

from search_eval.datasets import base
from pathlib import Path

def ensure_file_exists(path:Path):
    if not path.is_file():
        raise Exception(f"Invalid path {path}")
    return path

def import_from_disk(dirpath:Path, normalize=True):
    if not dirpath.is_dir():
        raise Exception(f"Invalid path {dirpath}")
    solution_file = ensure_file_exists(dirpath / Path("solution.csv"))
    train_file = ensure_file_exists(dirpath / Path("train.csv.zip"))
    test_file = ensure_file_exists(dirpath / Path("test.csv.zip"))
    product_descriptions = ensure_file_exists(dirpath / Path("product_descriptions.csv.zip"))
    attributes = ensure_file_exists(dirpath / Path("attributes.csv.zip"))

    df_solution =  pd.read_csv(str(solution_file))
    df_train = pd.read_csv(str(train_file), encoding="ISO-8859-1")
    df_test_ = pd.read_csv(str(test_file), encoding="ISO-8859-1")
    df_test = pd.merge(df_test_, df_solution, how='left', on='id')
    df_test = df_test[df_test.relevance != -1]
    df_pro_desc = pd.read_csv(str(product_descriptions))
    df_attr = pd.read_csv(attributes)
    df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
    
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

    df_all.to_csv(dirpath / Path("df_all.csv"))

    #build dataset object
    queries = df_all.search_term.values
    docs = df_all.product_title.values
    if normalize:
        relevance = (df_all.relevance.values - 1) / 2.0
    else:
        relevance = df_all.relevance.values - 1
    return queries, docs, relevance