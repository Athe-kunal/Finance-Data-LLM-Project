from src.vectorDatabase import get_all_docs
from collections import defaultdict
import concurrent.futures
import colbert
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from src.config import *
import os
from functools import lru_cache

os.environ['COLBERT_LOAD_TORCH_EXTENSION_VERBOSE'] = "True"

def build_index_name(split_docs):
    texts_dict = defaultdict(list)
    for doc in split_docs:
        text = doc.page_content
        metadata = doc.metadata
        if "filing_type" in metadata:
            filing_type = metadata['filing_type']
            texts_dict[filing_type].append(text)
        elif 'quarter' in metadata:
            quarter = metadata['quarter']
            texts_dict[quarter].append(text)
    return texts_dict

def build_index(ticker,year):
    split_docs,sec_form_names,earnings_call_quarter_vals,speakers_list_1,speakers_list_2,speakers_list_3, = get_all_docs(ticker,year)
    texts_dict = build_index_name(split_docs)
    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 768 # truncate passages at 300 tokens
    max_id = 10000
    with Run().context(RunConfig(nranks=1, experiment=EXPERIMENT_NAME)):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=8) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
                                                                                    # Consider larger numbers for small datasets.

        indexer = Indexer(checkpoint=COLBERT_CHECKPOINT, config=config)
        for name, text_list in texts_dict.items():
            index_name = f'SEC.Earningcalls.{ticker}.{year}.{name}.{nbits}bits'
            indexer.index(name=index_name, collection=text_list, overwrite=True)
    searcher_dict = {}

    for name,collection in texts_dict.items():
        index_name = f'SEC.Earningcalls.{ticker}.{year}.{name}.{nbits}bits'
        with Run().context(RunConfig(experiment=EXPERIMENT_NAME)):
            searcher = Searcher(index=index_name, collection=collection)
            searcher_dict[name] = searcher
    return searcher_dict

@lru_cache
def get_index(name,searcher_dict):
    return searcher_dict[name]

def query_data(query,name,searcher_dict):
    searcher = get_index(name,searcher_dict)
    results = searcher.search(query, k=COLBERT_RETURN_LIMIT)

    relevant_docs = ""
    for passage_id, _, _ in zip(*results):
        relevant_docs+=searcher.collection[passage_id]
        relevant_docs+="/n"
    return relevant_docs

