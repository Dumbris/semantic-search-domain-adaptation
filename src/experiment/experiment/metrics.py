import logging
from search_eval.datasets import base
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from search_eval.metrics import mapk, ndcg
import numpy as np

logger = logging.getLogger(__name__)

def calc_metrics(ds_test: base.Dataset, ds_pred: base.Dataset, metrics_k: List[int] = [3,5,10], need_sort=True) -> Dict:
    logger.info(f"Calculating metrics...")
    metrics = defaultdict(list)
    res = {}
    for (pred_query, pred_docs, pred_rels), (test_query, test_docs, test_rels) in zip(ds_pred, ds_test):
        assert pred_query == test_query
        if need_sort:
            pred_docs = pred_docs[np.argsort(pred_rels)[::-1]]
            test_docs = test_docs[np.argsort(test_rels)[::-1]]

        for k in metrics_k:
            metrics[f"apk@{k}"].append(mapk.apk(test_docs, pred_docs, k))
            metrics[f"ndcg@{k}"].append(ndcg.ndcg(test_rels, test_docs, pred_rels, pred_docs, k))

    for name, vals in metrics.items():
        val = np.mean(np.array(vals))
        res[name] = val
        logger.info(f"{name}: {val}")
    return res