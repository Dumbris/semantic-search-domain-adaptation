from rank_bm25 import BM25Okapi
import nltk
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
import numpy as np
from search_eval.models import base
from typing import List


nltk.download('punkt')
nltk.download('stopwords')

class BM25(base.Model):
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.corpus = None
        self.index = None

    def _stem_tokenize(self, text, remove_stopwords=True):
        tokens = [word for sent in nltk.sent_tokenize(text) \
                                            for word in nltk.word_tokenize(sent)]
        tokens = [word for word in tokens if word not in \
                nltk.corpus.stopwords.words('english')]
        return [self.stemmer.stem(word) for word in tokens]

    def build_index(self, docs: List[str]):
        docs = list(docs)
        tokenized_docs_list = map(self._stem_tokenize, docs)
        self.corpus = list(map(lambda x: ' '.join(x), tokenized_docs_list))
        self.index = BM25Okapi(self.corpus)

    def generate_candidates(self, query, k=10):
        tokenized_query = self._stem_tokenize(query, True)
        scores = self.index.get_scores(tokenized_query)
        idx = np.argsort(scores)[::-1][:k]
        #idx = np.argpartition(scores, -1*k)[-1*k:]
        return scores[idx], idx