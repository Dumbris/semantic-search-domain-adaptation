import pytest
import numpy as np
import sys
from search_eval.metrics.ndcg import ndcg

def test_a():
    ids_true = np.array([1001,1002,1003,1004,1005])
    y_true = np.array([2.3, 3, 0.5, 1, 8])

    ids_pred = np.array([2001,2002,1004,1003,1001])
    y_pred = np.array([0.5, 3.2, 1.5, 0.1, 0.3])

    assert ndcg(y_true, ids_true, y_pred, ids_pred) == 0.6406579541669738

def test_ideal():
    ids_true = np.array([1001,1002,1003,1004,1005])
    y_true = np.array([2.3, 3, 0.5, 1, 8])

    ids_pred = np.copy(ids_true)
    y_pred = np.copy(y_true)

    assert ndcg(y_true, ids_true, y_pred, ids_pred) == 1.0

def test_mismatch():
    ids_true = np.array([1001,1002,1003,1004,1005])
    y_true = np.array([2.3, 3, 0.5, 1, 8])

    ids_pred = np.array([2001,2002,2004,2003,2001])
    y_pred = np.array([0.5, 3.2, 1.5, 0.1, 0.3])

    assert ndcg(y_true, ids_true, y_pred, ids_pred) == 0.0