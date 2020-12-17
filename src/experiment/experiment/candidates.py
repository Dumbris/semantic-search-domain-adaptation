import logging
from search_eval.datasets import base

logger = logging.getLogger(__name__)


def get_new_candidates(index, ds_test, encoded_queries, k:int=50):
    candidates_pairs = []
    for (score, idx), query in zip(index.generate_candidates(encoded_queries, k), ds_test.queries_uniq):
        for doc_id, doc_relevancy in zip(ds_test.docs[idx], score):
            candidates_pairs.append((query, doc_id, doc_relevancy))
    
    assert len(candidates_pairs) == k*len(ds_test.queries_uniq)

    candidates_queries, candidates_docs, candidates_scores = zip(*candidates_pairs)

    return base.Dataset(candidates_queries, candidates_docs, candidates_scores)

#docs_corpus[ds.docs].tolist()
#ds.docs[idx]
def get_new_candidates2(index, ds_test, queries_corpus, docs_corpus, vectorizer, cfg, k:int=50):
    #Init index class
    index = ann.HNSWIndex(cfg.index.ann) #TODO: remove this dependency
    logger.info("Encode docs...")
    vectorized_docs = vectorizer.encode(docs_corpus[ds_test.docs], show_progress_bar=False, convert_to_numpy=True)
    logger.info("Indexing docs...")
    index.build(vectorized_docs)

    logger.info("Encode queries...")
    queries_list = queries_corpus[ds_test.queries_uniq].tolist()
    vectorized_queries = vectorizer.encode(queries_list, show_progress_bar=False, convert_to_numpy=True)
    #Reranking
    logger.info("Generate candidates for reranking...")
    candidates_pairs = []
    for (score, idx), query in zip(index.generate_candidates(vectorized_queries, k), ds_test.queries_uniq):
        for doc_id, doc_relevancy in zip(ds_test.docs[idx], score):
            candidates_pairs.append((query, doc_id, doc_relevancy))
    
    assert len(candidates_pairs) == k*len(ds_test.queries_uniq)

    candidates_queries, candidates_docs, candidates_scores = zip(*candidates_pairs)

    return base.Dataset(candidates_queries, candidates_docs, candidates_scores)