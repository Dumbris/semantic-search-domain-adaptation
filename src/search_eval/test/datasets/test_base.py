import pytest
import numpy as np
from search_eval.datasets.base import Dataset

def test_getitem():
    queries = np.array([101,101,101,102,102,103,103])
    docs = np.array([201,202,203,204,205,206,207])
    rels = np.array([0.3,2,1,0.4,3,1,1])
    ds = Dataset(queries, docs, rels)
    assert len(ds) == 3
    _idx, _docs, _ = ds[102]
    assert _idx == 102
    np.testing.assert_equal(_docs, np.array([204,205]))

def test_sample():
    queries = np.array([101,101,101,102,102,103,103])
    docs = np.array([201,202,203,204,205,206,207])
    rels = np.array([0.3,2,1,0.4,3,1,1])
    ds = Dataset(queries, docs, rels)
    assert len(ds) == 3
    sample1 = ds.sample(None)
    assert len(sample1) == 3
    sample2 = ds.sample(2)
    assert len(sample2) == 2
    sample3 = ds.sample(0.01)
    assert len(sample3) == 1