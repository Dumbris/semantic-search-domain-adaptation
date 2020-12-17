import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
from experiment.encoders.base import Encoder
from typing import List
import logging
logger = logging.getLogger(__name__)

batch_size = 2000
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

class Uset(Encoder):
    def __init__(self, cfg=None, name:str=module_url):
        logger.info(f"Loading {name} model...")
        self.name = name
        self.embed_fn = hub.load(module_url)

    def _get_emb_batched(self, arr):
        return np.concatenate([self.embed_fn(arr[i:i + batch_size]).numpy() for i in range(0, len(arr), batch_size)])

    def encode(self, docs:List[str]):
        return self._get_emb_batched(docs)