"""Oracle model. Just memorise all data and use it for answers.
"""

from search_eval.models.base import Model
from search_eval.datasets.base import Dataset
import numbers

class Oracle(Model):
    def __init__(self):
        self.ds_test =  None

    def generate_candidates(self, query, k=10):
        if not self.ds_test:
            raise Exception("You need to build index first, using build_index() method")
        if not isinstance(query, numbers.Integral): 
            raise Exception("Parameter query for Oracle must be index of question with numeric type")
        _, docs, rels = self.ds_test[query]
        rel_ids = rels.argsort()[::-1]
        return rels[rel_ids], docs[rel_ids]

    def build_index(self, ds_test:Dataset):
        self.ds_test = ds_test