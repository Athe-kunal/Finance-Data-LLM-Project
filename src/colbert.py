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
import torch
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter

os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "True"


def build_index_name(split_docs):
    texts_dict = defaultdict(list)
    for doc in split_docs:
        text = doc.page_content
        metadata = doc.metadata
        if "filing_type" in metadata:
            filing_type = metadata["filing_type"]
            texts_dict[filing_type].append(text)
        elif "quarter" in metadata:
            quarter = metadata["quarter"]
            texts_dict[quarter].append(text)
    return texts_dict


def build_index(ticker, year):
    (
        docs,
        sec_form_names,
        earnings_call_quarter_vals,
        speakers_list_1,
        speakers_list_2,
        speakers_list_3,
    ) = get_all_docs(ticker, year)
    texts_dict = build_index_name(docs)
    nbits = NBITS  # encode each dimension with 2 bits
    doc_maxlen = DOC_MAXLEN  # truncate passages at 300 tokens
    max_id = 10000
    with Run().context(
        RunConfig(nranks=1, experiment=EXPERIMENT_NAME)
    ):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(
            doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4
        )  # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
        # Consider larger numbers for small datasets.

        indexer = Indexer(checkpoint=COLBERT_CHECKPOINT, config=config)
        for name, text_list in texts_dict.items():
            index_name = f"SEC.Earningcalls.{ticker}.{year}.{name}.{nbits}bits"
            indexer.index(name=index_name, collection=text_list, overwrite=True)
    searcher_dict = {}

    for name, collection in texts_dict.items():
        index_name = f"SEC.Earningcalls.{ticker}.{year}.{name}.{nbits}bits"
        with Run().context(RunConfig(experiment=EXPERIMENT_NAME)):
            searcher = Searcher(index=index_name, collection=collection)
            searcher_dict[name] = searcher
    return searcher_dict


def build_index_all(ticker, year):
    docs, sec_form_names, earnings_call_quarter_vals, _, _, _ = get_all_docs(
        ticker, year
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=COLBERT_CHUNK_SIZE,
        chunk_overlap=COLBERT_CHUNK_OVERLAP,
        length_function=len,
        # is_separator_regex = False,
    )
    split_docs = text_splitter.split_documents(docs)
    quarter_forms_dict = {k: {} for k in sec_form_names + earnings_call_quarter_vals}
    sentences = []
    for idx, doc in enumerate(split_docs):
        if doc.page_content == "":
            continue
        doc_metadata = doc.metadata
        sentences.append(doc.page_content)
        if "quarter" in doc_metadata:
            # quarter_forms_dict[doc_metadata["quarter"]].append((idx, doc_metadata))
            quarter_forms_dict[doc_metadata["quarter"]].update({str(idx):doc_metadata})
        elif "filing_type" in doc_metadata:
            quarter_forms_dict[doc_metadata["filing_type"]].update({str(idx):doc_metadata})

    nbits = NBITS  # encode each dimension with 2 bits
    doc_maxlen = DOC_MAXLEN  # truncate passages at 300 tokens
    max_id = 10000
    with Run().context(
        RunConfig(nranks=1, experiment=EXPERIMENT_NAME)
    ):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(
            doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4
        )  # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
        # Consider larger numbers for small datasets.

        indexer = Indexer(checkpoint=COLBERT_CHECKPOINT, config=config)
        index_name = f"SEC.Earningcalls.{ticker}.{year}.{nbits}bits"
        indexer.index(name=index_name, collection=sentences, overwrite=True)

    index_name = f"SEC.Earningcalls.{ticker}.{year}.{nbits}bits"
    with Run().context(RunConfig(experiment=EXPERIMENT_NAME)):
        searcher = Searcher(index=index_name, collection=sentences)
    return searcher, quarter_forms_dict


@lru_cache
def get_index(name, searcher_dict):
    return searcher_dict[name]


def query_data(query, name, searcher_dict):
    searcher = get_index(name, searcher_dict)
    results = searcher.search(query, k=COLBERT_RETURN_LIMIT)

    relevant_docs = ""
    for passage_id, _, _ in zip(*results):
        relevant_docs += searcher.collection[passage_id]
        relevant_docs += "/n"
    return relevant_docs


def query_data_all(query: str, searcher, quarter_or_form_name: str, quarter_forms_dict):
    # relevant_ids = torch.tensor(quarter_forms_dict[quarter_or_form_name])
    required_quarter_form_dict = quarter_forms_dict[quarter_or_form_name]
    relevant_ids = torch.tensor([int(i) for i in required_quarter_form_dict.keys()]).to("cuda:0")
    # print(relevant_ids.dtype,relevant_ids)
    results = searcher.search(
        query,
        k=COLBERT_RETURN_LIMIT,
        filter_fn=lambda pids: torch.tensor(
            [pid for pid in pids if pid in relevant_ids],dtype=torch.int32).to("cuda:0"))

    relevant_docs = ""
    if quarter_or_form_name.startswith("Q"):
      speaker_dict = {}
      print(*results)
      for passage_id, _, _ in zip(*results):
        metadata = required_quarter_form_dict[str(passage_id)]
        speaker = metadata['speaker']
        if speaker not in speaker_dict: speaker_dict[speaker]=""
        speaker_dict[speaker]+=searcher.collection[passage_id]
      for speaker,text in speaker_dict.items():
        relevant_docs+=speaker+": "
        relevant_docs+=text + "\n\n"
    elif quarter_or_form_name.startswith("10"):
      section_dict = {}
    #   print(*results)
      for passage_id, _, _ in zip(*results):
        print(passage_id,searcher.collection[passage_id])
        metadata = required_quarter_form_dict[str(passage_id)]
        section = metadata['sectionName']
        if section not in section_dict: section_dict[section]=""
        section_dict[section]+=searcher.collection[passage_id]
      for section,text in section_dict.items():
        relevant_docs+=section+": "
        relevant_docs+=text + "\n\n"
    return relevant_docs
