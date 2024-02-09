# AIM OF THE FILE

"""
Give an API request to all the backend requests
"""
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel
from src.vectorDatabase import create_database
from src.queryDatabase import query_database_earnings_call, query_database_sec
import os
from src.chat_earnings_call import get_openai_answer_earnings_call
from src.chat_sec import get_openai_answer_sec

app = FastAPI()

@lru_cache
def get_settings():
    return QdrantSettings()

@lru_cache
def get_client_encoder(reset:bool=False):
    return os.environ["QDRANT"], os.environ["ENCODER"]

@app.get("data/{ticker}/{year}")
async def ticker_year(ticker:str,year:int):
    (
        qdrant_client,
        encoder,
        speakers_list_1,
        speakers_list_2,
        speakers_list_3,
        sec_form_names,
        earnings_call_quarter_vals,
    ) = create_database(ticker=ticker, year=year)
    os.environ["QDRANT"] = qdrant_client
    os.environ["ENCODER"] = encoder
    os.environ["SPEAKER_LIST_1"] = speakers_list_1
    os.environ["SPEAKER_LIST_2"] = speakers_list_2
    os.environ["SPEAKER_LIST_3"] = speakers_list_3
    os.environ["SEC_FORM_NAMES"] = sec_form_names
    os.environ["EARNING_CALLS_QUARTER_VALS"] = earnings_call_quarter_vals

@app.get("Earnings/{question}/{quarter}")
async def earnings_chat(question:str,quarter:str):
    # qdrant_client,encoder = get_client_encoder()
    qdrant_client = os.environ["QDRANT"]
    encoder = os.environ["ENCODER"]
    if quarter == "Q1":
        speakers_list = os.environ["speaker_list_1"]
    elif quarter == "Q2":
        speakers_list = os.environ["speaker_list_2"]
    elif quarter == "Q3":
        speakers_list = os.environ["speaker_list_3"]
    relevant_text = query_database_earnings_call(
        question, quarter, qdrant_client, encoder, speakers_list
    )
    res = get_openai_answer_earnings_call(question, relevant_text)

    return res, relevant_text

@app.get("SEC/{question}/{doc_name}")
async def sec_chat(question:str,doc_name:str):
    # qdrant_client,encoder = get_client_encoder()
    qdrant_client = os.environ["QDRANT"]
    encoder = os.environ["ENCODER"]
    sec_form_names = os.environ["SEC_FORM_NAMES"]
    assert doc_name in sec_form_names, f"The document name should be in the list {sec_form_names}"

    relevant_text = query_database_sec(question,qdrant_client,encoder,doc_name)

    res = get_openai_answer_sec(question,relevant_text)

    return res, relevant_text
    

