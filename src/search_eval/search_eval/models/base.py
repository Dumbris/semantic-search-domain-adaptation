"""Module contains super class for all ranking models
"""

import numpy as np

from pathlib import Path
import numbers

class Model:
    def __init__(self):
        pass

    def generate_candidates(self, query, k=10):
        pass

    def build_index(self):
        pass
