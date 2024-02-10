from pydantic_settings import BaseSettings
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List


class QdrantSettings(BaseSettings):
    qdrant_client: QdrantClient
    encoder: SentenceTransformer
    speaker_list_1: List[str]
    speaker_list_2: List[str]
    speaker_list_3: List[str]
    sec_form_names: List[str]
    earnings_call_quarter_vals: List[str]
