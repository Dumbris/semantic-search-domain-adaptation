from typing import List
from sentence_transformers import SentenceTransformer
from experiment.encoders.base import Encoder

class SentTrans(Encoder):
    def __init__(self, cfg, name:str="SentenceTransformer"):
        self.cfg = cfg
        self.name = name
        self.encoder = SentenceTransformer(cfg.base_model)
    
    def encode(self, docs:List, show_progress_bar=False, convert_to_numpy=True):
        return self.encoder.encode(docs, show_progress_bar=show_progress_bar, convert_to_numpy=convert_to_numpy)