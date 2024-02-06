# AIM OF THE FILE

"""
Give an API request to all the backend requests
"""

from fastapi import FastAPI
from pydantic import BaseModel
from src.vectorDatabase import create_database
from src.queryDatabase import query_database_earnings_call, query_database_sec

app = FastAPI()
