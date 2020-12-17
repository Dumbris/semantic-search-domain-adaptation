"""Base class for index 
"""

from typing import List

class BaseIndex:
    def __init__(self, cfg, name:str="index"):
        self.cfg = cfg
        self.name = name
    
    def encode(self, docs:List[str]):
        raise Exception("Not implemented method.")

    def build(self, docs:List):
        raise Exception("Not implemented method.")

    def generate_candidates(self, encoded_queries: List, k=10):
        raise Exception("Not implemented method.")