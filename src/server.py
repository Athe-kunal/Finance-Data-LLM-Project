# AIM OF THE FILE

"""
Give an API request to all the backend requests
"""

from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel
from src.vectorDatabase import create_database
from src.queryDatabase import query_database_earnings_call, query_database_sec
from .settings import QdrantSettings

app = FastAPI()

@lru_cache
def get_settings():
    return QdrantSettings()

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

# @app.get("Earnings/{question}/{quarter}")
# async def earnings_chat(question:str,quarter:str):
