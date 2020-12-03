"""Module for run queries with search models
    algo:
    for each query
        get candidates from model
        calc metrics using matched docs and docs from judgements list
"""
import sys
from pathlib import Path
import numpy as np
from search_eval.datasets import base
from search_eval.models import bm25
from search_eval.metrics import mapk
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm import tqdm
from search_eval.progressparallel import ProgressParallel
from search_eval import homedepot
import logging

logger = logging.getLogger(__name__)