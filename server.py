# AIM OF THE FILE

"""
Give an API request to all the backend requests
"""
from functools import lru_cache
from fastapi import FastAPI
# from fastapi_sessions import SessionCookie
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from src.vectorDatabase import create_database
from src.queryDatabase import query_database_earnings_call, query_database_sec
import os
from src.chat_earnings_call import get_openai_answer_earnings_call
from src.chat_sec import get_openai_answer_sec
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List
import contextvars

app = FastAPI()

# class SessionData(BaseModel):
#     class Config:
#         arbitrary_types_allowed = True
#     qdrant_client: QdrantClient = None
#     encoder: SentenceTransformer = None
#     speaker_list_1: List[str] = []
#     speaker_list_2: List[str] = []
#     speaker_list_3: List[str] = []
#     sec_form_name: List[str] = []
#     earnings_call_quarter_vals: List[str] = []
qdrant_client: QdrantClient = contextvars.ContextVar("qdrant_client", default = None)
encoder: SentenceTransformer = contextvars.ContextVar("encoder",default=None)
speakers_list_1: List[str] = contextvars.ContextVar("speaker_list_1",default=[])
speakers_list_2: List[str] = contextvars.ContextVar("speaker_list_2",default=[])
speakers_list_3: List[str] = contextvars.ContextVar("speaker_list_3",default=[])
sec_form_names: List[str] = contextvars.ContextVar("sec_form_names",default=[])
earnings_call_quarter_vals: List[str] = contextvars.ContextVar("earnings_call_quarter_vals",default=[])


@app.get("/data/{ticker}/{year}")
async def ticker_year(ticker:str,year:int):
    print(ticker,year)
    (
        qdrant_client_,
        encoder_,
        speakers_list_1_,
        speakers_list_2_,
        speakers_list_3_,
        sec_form_names_,
        earnings_call_quarter_vals_,
    ) = create_database(ticker=ticker, year=year)
    
    qdrant_client.set(qdrant_client_)
    encoder.set(encoder_)
    speakers_list_1.set(speakers_list_1_)
    speakers_list_1.set(speakers_list_2_)
    speakers_list_1.set(speakers_list_3_)
    sec_form_names.set(sec_form_names_)
    earnings_call_quarter_vals.set(earnings_call_quarter_vals_)

@app.get("/Earnings/{question}/{quarter}")
async def earnings_chat(question:str,quarter:str):
    # qdrant_client = os.environ["QDRANT"]
    # encoder = os.environ["ENCODER"]
    qdrant_client_,encoder_ = qdrant_client.get(),encoder.get()
    if quarter == "Q1":
        speakers_list = speakers_list_1.get()
    elif quarter == "Q2":
        speakers_list = speakers_list_2.get()
    elif quarter == "Q3":
        speakers_list = speakers_list_3.get()
    relevant_text = query_database_earnings_call(
        question, quarter, qdrant_client_, encoder_, speakers_list
    )
    res = get_openai_answer_earnings_call(question, relevant_text)

    return res, relevant_text

@app.get("/SEC/{question}/{doc_name}")
async def sec_chat(question:str,doc_name:str):
    # qdrant_client,encoder = get_client_encoder()
    qdrant_client,encoder = qdrant_client.get(),encoder.get()
    # qdrant_client = os.environ["QDRANT"]
    # encoder = os.environ["ENCODER"]
    sec_form_names = sec_form_names.get()
    assert doc_name in sec_form_names, f"The document name should be in the list {sec_form_names}"

    relevant_text = query_database_sec(question,qdrant_client,encoder,doc_name)

    res = get_openai_answer_sec(question,relevant_text)

    return res, relevant_text
    

