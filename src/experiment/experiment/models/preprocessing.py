"""Module for text tokenization
"""

import spacy

def lemmatize_pipe(doc):
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if (tok.is_digit or tok.is_alpha) and not tok.is_stop] 
    return lemma_list

class Preprocess:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def preprocess_pipe(self, texts, batch_size=500):
        preproc_pipe = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            preproc_pipe.append(lemmatize_pipe(doc))
        return preproc_pipe