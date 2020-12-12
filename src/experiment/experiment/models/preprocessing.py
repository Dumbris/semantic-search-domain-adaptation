"""Module for text tokenization
"""

import spacy

spacy_model_name = 'en_core_web_sm'

def lemmatize_pipe(doc):
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if (tok.is_digit or tok.is_alpha) and not tok.is_stop] 
    return lemma_list

class Preprocess:
    def __init__(self):
        if not spacy.util.is_package(spacy_model_name):
            spacy.cli.download(spacy_model_name)
        nlp = spacy.load(spacy_model_name)
        self.nlp = spacy.load(spacy_model_name, disable=['tagger', 'parser', 'ner'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def preprocess_pipe(self, texts, batch_size=500):
        preproc_pipe = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            preproc_pipe.append(lemmatize_pipe(doc))
        return preproc_pipe