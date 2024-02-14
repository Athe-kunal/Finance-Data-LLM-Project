# AIM OF THE FILE

"""
Give an API request to all the backend requests
"""
from fastapi import FastAPI
from src.vectorDatabase import create_database
from src.queryDatabase import query_database_earnings_call, query_database_sec
import os
from src.chat_earnings_call import get_openai_answer_earnings_call
from src.chat_sec import get_openai_answer_sec
from dotenv import load_dotenv
import openai

try:
    load_dotenv()
except:
    pass
openai.api_key = os.environ["OPENAI_API_KEY"]

app = FastAPI()


@app.get("/data/{ticker}/{year}")
async def ticker_year(ticker: str, year: int):
    # print(ticker, year)
    (
        qdrant_client_,
        encoder_,
        speakers_list_1_,
        speakers_list_2_,
        speakers_list_3_,
        speakers_list_4_,
        sec_form_names_,
        earnings_call_quarter_vals_,
    ) = create_database(ticker=ticker, year=year)

    global qdrant_client
    qdrant_client = qdrant_client_
    global encoder
    encoder = encoder_
    global speakers_list_1
    speakers_list_1 = speakers_list_1_
    global speakers_list_2
    speakers_list_2 = speakers_list_2_
    global speakers_list_3
    speakers_list_3 = speakers_list_3_
    global speakers_list_4
    speakers_list_4 = speakers_list_4_
    global sec_form_names
    sec_form_names = sec_form_names_
    global earnings_call_quarter_vals
    earnings_call_quarter_vals = earnings_call_quarter_vals_


@app.get("/Earnings/{question}/{quarter}")
async def earnings_chat(question: str, quarter: str):
    if quarter == "Q1":
        speakers_list = speakers_list_1
    elif quarter == "Q2":
        speakers_list = speakers_list_2
    elif quarter == "Q3":
        speakers_list = speakers_list_3
    elif quarter == "Q4":
        speakers_list = speakers_list_4
    relevant_text = query_database_earnings_call(
        question, quarter, qdrant_client, encoder, speakers_list
    )
    res = get_openai_answer_earnings_call(question, relevant_text)

    return res, relevant_text


@app.get("/SEC/{question}/{doc_name}")
async def sec_chat(question: str, doc_name: str):
    assert (
        doc_name in sec_form_names
    ), f"The document name should be in the list {sec_form_names}"

    relevant_text = query_database_sec(question, qdrant_client, encoder, doc_name)

    res = get_openai_answer_sec(question, relevant_text)

    return res, relevant_text
