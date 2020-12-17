"""Module for text tokenization
"""
from experiment.encoders.base import Encoder
import spacy



def lemmatize_pipe(doc):
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if (tok.is_digit or tok.is_alpha) and not tok.is_stop] 
    return lemma_list

class Tokenizer(Encoder):
    def __init__(self, cfg, name="Tokenizer"):
        self.cfg = cfg
        self.name = name
        self.spacy_model_name = self.cfg.spacy_model_name
        if not spacy.util.is_package(self.spacy_model_name):
            spacy.cli.download(self.spacy_model_name)
        self.nlp = spacy.load(self.spacy_model_name, disable=['tagger', 'parser', 'ner'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def encode(self, texts, batch_size=500):
        preproc_pipe = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            preproc_pipe.append(lemmatize_pipe(doc))
        return preproc_pipe