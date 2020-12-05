"""DCG and NDCG metrics
https://nbviewer.jupyter.org/github/ogrisel/notebooks/blob/master/Learning%20to%20Rank.ipynb
"""

import numpy as np

def _dcg(relevances, rank=10):
    """Discounted cumulative gain at rank (DCG)"""
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)
 
 
def _ndcg(relevances, rank=10):
    """Normalized discounted cumulative gain (NDGC)"""
    relevances = np.asarray(relevances)
    best_dcg = _dcg(np.sort(relevances)[::-1], rank)
    if best_dcg == 0:
        return 0.

    return _dcg(relevances, rank) / best_dcg

def ndcg(y_true, ids_true, y_pred, ids_pred, rank=10):
    ranked_ids_true = ids_true[np.argsort(y_true)[::-1]]
    ranked_ids_pred = ids_pred[np.argsort(y_pred)[::-1]]
    ranked_y_pred = np.sort(y_pred)[::-1]
    #np.where(np.in1d(ranked_ids_pred, ranked_ids_true), ranked_ids_pred, 0), 
    relevances = np.where(np.in1d(ranked_ids_pred, ranked_ids_true), ranked_y_pred, 0)
    return _ndcg(relevances, rank) 