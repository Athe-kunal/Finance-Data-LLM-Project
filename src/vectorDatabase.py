from src.utils import get_earnings_transcript
import re
from langchain.schema import Document
from src.config import *
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
from qdrant_client.models import VectorParams, Distance
import openai
# from openai. import OpenAI
# from qdrant_client.local import
import os
import json
from src.secData import sec_main
from tenacity import RetryError
from dotenv import load_dotenv

load_dotenv(".env")

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
        span_range = match_.span()
        ranges.append(span_range)
        speakers_list.append(match_.group())
    speakers_list = [clean_speakers(sl) for sl in speakers_list]

    for idx, speaker in enumerate(speakers_list[:-1]):
        start_range = ranges[idx][1]
        end_range = ranges[idx + 1][0]
        speaker_text = content[start_range + 1 : end_range]

        docs.append(
            Document(
                page_content=speaker_text,
                metadata={"speaker": speaker, "quarter": quarter},
            )
        )

    docs.append(
        Document(
            page_content=content[ranges[-1][1] :],
            metadata={"speaker": speakers_list[-1], "quarter": quarter},
        )
    )
    return docs, speakers_list


def get_all_docs(ticker: str, year: int):
    docs = []
    earnings_call_quarter_vals = []
    print("Earnings Call Q1")
    try:
        docs, speakers_list_1 = get_earnings_all_quarters_data(docs, "Q1", ticker, year)
        earnings_call_quarter_vals.append("Q1")
    except RetryError:
        print(f"Don't have the data for Q1")
        speakers_list_1 = []

    print("Earnings Call Q2")
    try:
        docs, speakers_list_2 = get_earnings_all_quarters_data(docs, "Q2", ticker, year)
        earnings_call_quarter_vals.append("Q2")
    except RetryError:
        print(f"Don't have the data for Q2")
        speakers_list_2 = []
    print("Earnings Call Q3")
    try:
        docs, speakers_list_3 = get_earnings_all_quarters_data(docs, "Q3", ticker, year)
        earnings_call_quarter_vals.append("Q3")
    except RetryError:
        print(f"Don't have the data for Q3")
        speakers_list_3 = []
    print("Earnings Call Q4")
    try:
        docs, speakers_list_4 = get_earnings_all_quarters_data(docs, "Q4", ticker, year)
        earnings_call_quarter_vals.append("Q4")
    except RetryError:
        print(f"Don't have the data for Q4")
        speakers_list_4 = []
    print("SEC")
    section_texts, sec_form_names = sec_main(ticker, year)

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
    return (
        docs,
        sec_form_names,
        earnings_call_quarter_vals,
        speakers_list_1,
        speakers_list_2,
        speakers_list_3,
        speakers_list_4,
    )


def create_database(ticker: str, year: int, curr_year_bool: bool = False):
    """Build the database to query from it

    Args:
        ticker (str): The ticker of the company
        year (int): The year of the earnings call
        curr_year (bool): If current year or not
    """
    database_folder = f"{DATABASE_FOLDER}/{ticker}-{year}-db"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = SentenceTransformer(
        ENCODER_NAME, device=device, trust_remote_code=True
    )  # or device="cpu" if you don't have a GPU
    openai_client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
    if os.path.exists(database_folder) and not curr_year_bool:
        with open(
            os.path.join(database_folder, "application_metadata.json"), "r"
        ) as openfile:
            application_metadata = json.load(openfile)
        earnings_call_quarter_vals = application_metadata["earnings_call_quarter_vals"]
        speakers_list_1 = application_metadata["speakers_list_1"]
        speakers_list_2 = application_metadata["speakers_list_2"]
        speakers_list_3 = application_metadata["speakers_list_3"]
        speakers_list_4 = application_metadata["speakers_list_4"]
        sec_form_names = application_metadata["sec_form_names"]

        qdrant_client = QdrantClient(path=database_folder)
        # qdrant_client.create_collection(
        #     collection_name=COLLECTION_NAME,
        #     vectors_config=VectorParams(
        #         size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE
        #     ),
        #     init_from=models.InitFrom(collection=COLLECTION_NAME)
        # )
        return (
            qdrant_client,
            encoder,
            speakers_list_1,
            speakers_list_2,
            speakers_list_3,
            speakers_list_4,
            sec_form_names,
            earnings_call_quarter_vals,
        )
    (
        docs,
        sec_form_names,
        earnings_call_quarter_vals,
        speakers_list_1,
        speakers_list_2,
        speakers_list_3,
        speakers_list_4,
    ) = get_all_docs(ticker=ticker, year=year)
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
        unrolled_dict["text"] = doc.page_content
        for k, v in doc.metadata.items():
            unrolled_dict[k] = v

        split_docs_qdrant.append(unrolled_dict)
    # qdrant_client = QdrantClient("http://localhost:6333")
    qdrant_client = QdrantClient(path=database_folder)

    if curr_year_bool:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                # size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE
                size=1536,
                distance=Distance.COSINE,
            ),
        )
    else:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                # size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE
                size=1536,
                distance=Distance.COSINE,
            ),
        )

    qdrant_client.upload_records(
        collection_name=COLLECTION_NAME,
        records=[
            models.Record(
                # id=idx, vector=encoder.encode(doc["text"]).tolist(), payload=doc
                id=idx,
                vector=openai_client.embeddings.create(
                    input=doc["text"], model=OPENAI_EMBEDDING_MODEL
                )
                .data[0]
                .embedding,
                payload=doc,
            )
            for idx, doc in enumerate(split_docs_qdrant)
        ],
    )

    metadata_dict = {
        "speakers_list_1": speakers_list_1,
        "speakers_list_2": speakers_list_2,
        "speakers_list_3": speakers_list_3,
        "speakers_list_4": speakers_list_4,
        "sec_form_names": sec_form_names,
        "earnings_call_quarter_vals": earnings_call_quarter_vals,
    }

    with open(
        f"{DATABASE_FOLDER}/{ticker}-{year}-db/application_metadata.json", "w"
    ) as f:
        json.dump(metadata_dict, f)

    return (
        qdrant_client,
        encoder,
        speakers_list_1,
        speakers_list_2,
        speakers_list_3,
        speakers_list_4,
        sec_form_names,
        earnings_call_quarter_vals,
    )
