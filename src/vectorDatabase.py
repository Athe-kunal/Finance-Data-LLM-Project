from utils import get_earnings_transcript
import re
from langchain.schema import Document
from config import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pandas as pd
from tqdm.notebook import tqdm
import torch
from qdrant_client import models, QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    Field,
    FieldCondition,
    MatchAny,
    Filter,
    Match,
)
from secData import sec_main


def clean_speakers(speaker):
    speaker = re.sub("\n", "", speaker)
    speaker = re.sub(":", "", speaker)
    return speaker


def get_earnings_all_quarters_data(docs, quarter: str, ticker: str, year: int):
    resp_dict = get_earnings_transcript(quarter, ticker, year)

    content = resp_dict["content"]
    pattern = re.compile(r"\n(.*?):")
    matches = pattern.finditer(content)

    speakers_list = []
    ranges = []
    for match_ in matches:
        # print(match.span())
        span_range = match_.span()
        # first_idx = span_range[0]
        # last_idx = span_range[1]
        ranges.append(span_range)
        speakers_list.append(match_.group())
    speakers_list = [clean_speakers(sl) for sl in speakers_list]

    for idx, speaker in enumerate(speakers_list[:-1]):
        start_range = ranges[idx][1]
        end_range = ranges[idx + 1][0]
        speaker_text = content[start_range + 1 : end_range]

        docs.append(Document(page_content=speaker_text, metadata={"speaker": speaker,"quarter":quarter}))

    docs.append(
        Document(
            page_content=content[ranges[-1][1] :],
            metadata={"speaker": speakers_list[-1],"quarter":quarter},
        )
    )
    return docs, speakers_list


def create_database(ticker: str, year: int):
    """Build the database to query from it

    Args:
        quarter (str): The quarter of the earnings call
        ticker (str): The ticker of the company
        year (int): The year of the earnings call
    """

    docs = []
    print("Earnings Call Q1")
    docs,speakers_list_1 = get_earnings_all_quarters_data(docs, "Q1", ticker, year)
    print("Earnings Call Q2")
    docs,speakers_list_2 = get_earnings_all_quarters_data(docs, "Q2", ticker, year)
    print("Earnings Call Q3")
    docs,speakers_list_3 = get_earnings_all_quarters_data(docs, "Q3", ticker, year)

    print("SEC")
    section_texts = sec_main(ticker, year)

    for filings in section_texts:
        texts_dict = filings[-1]

        for section_name, text in texts_dict.items():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "accessionNumber": filings[0],
                        "filing_type": filings[1],
                        "filingDate": filings[2],
                        "reportDate": filings[3],
                        "sectionName": section_name,
                    },
                )
            )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # is_separator_regex = False,
    )
    split_docs = text_splitter.split_documents(docs)
    # print(split_docs_qdrant)
    # return
    split_docs_qdrant = []
    for doc in split_docs:
        unrolled_dict = {}
        unrolled_dict['text'] = doc.page_content
        for k,v in doc.metadata.items():
            unrolled_dict[k] = v

        split_docs_qdrant.append(unrolled_dict)
    qdrant_client = QdrantClient("http://localhost:6333")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = SentenceTransformer(
        ENCODER_NAME, device=device
    )  # or device="cpu" if you don't have a GPU

    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE
        ),
    )

    qdrant_client.upload_records(
        collection_name=COLLECTION_NAME,
        records=[
            models.Record(
                id=idx, vector=encoder.encode(doc["text"]).tolist(), payload=doc
            )
            for idx, doc in enumerate(split_docs_qdrant)
        ],
    )

    return qdrant_client, encoder, speakers_list_1, speakers_list_2, speakers_list_3


def query_database_earnings_call(question: str, quarter:str, qdrant_client, encoder, speakers_list):
    req_speaker_list = []
    for sl in speakers_list:
        if sl in question:
            req_speaker_list.append(sl)

    if len(req_speaker_list) == 0:
        req_speaker_list = speakers_list

    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=encoder.encode(question).tolist(),
        limit=EARNINGS_CALL_RETURN_LIMIT,
        # query_filter=models.Filter(
        #     must=[
        #         models.FieldCondition(
        #             key="speaker",
        #             match=models.MatchAny(
        #                 any=req_speaker_list,
        #             ),
        #         )
        #     ]
        # ),
        query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="speaker",
                match=models.MatchAny(any=req_speaker_list),
            ),
            models.FieldCondition(
                key="quarter",
                match=models.MatchValue(value=quarter),
                ),
            ]
        ),
        search_params=models.SearchParams(hnsw_ef=256, exact=True),
    )

    relevant_docs = []
    for hit in hits:
        relevant_docs.append(hit.payload)

    relevant_docs_speaker_list = []
    for rd in relevant_docs:
        curr_speaker = rd["speaker"]
        if curr_speaker not in relevant_docs_speaker_list:
            relevant_docs_speaker_list.append(rd["speaker"])

    relevant_speaker_dict = {k: "" for k in relevant_docs_speaker_list}
    for rd in relevant_docs:
        relevant_speaker_dict[rd["speaker"]] += rd["text"]

    relevant_speaker_text = ""
    for speaker, text in relevant_speaker_dict.items():
        relevant_speaker_text += speaker + ": "
        relevant_speaker_text += text + "\n\n"

    return relevant_speaker_text

def query_database_sec(question: str, qdrant_client, encoder, search_form: str):

    assert search_form in ["10-K","10-Q1","10-Q2","10-Q3"], f'The search form type should be in ["10-K","10-Q1","10-Q2","10-Q3"]'

    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=encoder.encode(question).tolist(),
        limit=SEC_DOCS_RETURN_LIMIT,
        query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="filing_type",
                match=models.MatchValue(value=search_form),
            )
            ]
        ),
        search_params=models.SearchParams(hnsw_ef=256, exact=True),
        )

    releavnt_docs = [hit.payload for hit in hits]

    section_text_dict = {}

    for rd in releavnt_docs:
        section_name = rd['section_name']
        section_text_dict[section_name]+= rd["text"] + '. '
    relevant_docs_sentences = ""
    rag_sentence_docs_sentences = ""
    
    for sec_name,section_text in section_text_dict.items():
        relevant_docs_sentences+=sec_name+": "
        relevant_docs_sentences+=section_text
        relevant_docs_sentences+="\n\n"
        rag_sentence_docs_sentences+=section_text
        rag_sentence_docs_sentences+="\n\n"

    return rag_sentence_docs_sentences, relevant_docs_sentences