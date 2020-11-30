"""Module contains super class for all datasets
"""

import numpy as np

from pathlib import Path
import numbers

class Dataset:
    def __init__(self, queries, docs, relevance):
        self.queries = np.asarray(queries)
        self.docs = np.asarray(docs)
        self.relevance = np.asarray(relevance)
        if self.queries.shape[0] != self.docs.shape[0]: 
            raise Exception(f"Queries, Docs length mismatch {self.queries.shape[0]} != {self.docs.shape[0]}")
        self.queries_uniq, self.judgements = np.unique(self.queries, return_inverse=True)
        self.queries_idx = np.arange(self.queries_uniq.shape[0])
        self.docs_idx = np.arange(self.docs.shape[0]) #TODO: Do we need it? 

    def __repr__(self):
        return f"""queries\t{self.queries_uniq}\t{self.queries_uniq.shape}\n \
                judgements\t{self.judgements}\t{self.judgements.shape}
                relevance\t{self.relevance}\t{self.relevance.shape}
                docs\t{self.docs}\t{self.docs.shape}
                """

    @classmethod
    def load_from_cache(cls, filename:Path):
        npzfile = np.load(str(filename), allow_pickle=True)
        return cls(npzfile["queries"], npzfile["docs"], npzfile["relevance"])
        
    def save_to_cache(self, filename:Path):
        np.savez(str(filename), 
                    queries=self.queries, 
                    relevance=self.relevance, 
                    docs=self.docs)

    def split_train_test(self, n_test_groups=0.6):
        n_test_groups = self._get_n_split(n_test_groups)
        permutation = np.random.permutation(self.queries_idx)
        queries_indx_test = permutation[:n_test_groups]
        queries_indx_train = permutation[n_test_groups:]

        test_idx = np.flatnonzero(np.in1d(self.judgements, queries_indx_test))
        train_idx = np.flatnonzero(np.in1d(self.judgements, queries_indx_train))

        ds_test = Dataset(self.queries[test_idx], self.docs[test_idx], self.relevance[test_idx])
        ds_train = Dataset(self.queries[train_idx], self.docs[train_idx], self.relevance[train_idx])
        return ds_test, ds_train

    def _get_n_split(self, n_test_groups):
        #Calc split size
        if isinstance(n_test_groups, float):
            n_test_groups = int(n_test_groups * self.queries.shape[0])

        if not isinstance(n_test_groups, numbers.Integral): 
            raise Exception("n_test_groups must be int of float")
        return n_test_groups

    def __len__(self):
        return self.queries_uniq.shape[0]

    def __getitem__(self, idx):
        mask = self.judgements == idx
        return self.queries_uniq[idx], self.docs_idx[mask], self.relevance[mask]
