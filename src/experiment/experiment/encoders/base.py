"""Base class for encoders for indexer
"""

from typing import List

class Encoder:
    def __init__(self, cfg, name:str="encoder"):
        self.cfg = cfg
        self.name = name
    
    def encode(self, docs:List[str]):
        raise Exception("Not implemented method.")