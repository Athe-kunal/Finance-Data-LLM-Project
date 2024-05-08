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
from src.config import *
from collections import defaultdict


def query_database_earnings_call(
    question: str, quarter: str, qdrant_client, encoder, speakers_list
):
    req_speaker_list = []
    for sl in speakers_list:
        if sl in question:
            req_speaker_list.append(sl)

    if len(req_speaker_list) == 0:
        req_speaker_list = speakers_list

    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=encoder.encode(question).tolist(),
        # query_vector=encoder.embeddings.create(
        #     input=question, model=OPENAI_EMBEDDING_MODEL
        # )
        # .data[0]
        # .embedding,
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
    assert search_form in [
        "10-K",
        "10-Q1",
        "10-Q2",
        "10-Q3",
        "10-Q4",
    ], f'The search form type should be in ["10-K","10-Q1","10-Q2","10-Q3"]'

    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=encoder.encode(question).tolist(),
        # query_vector=encoder.embeddings.create(
        #     input=question, model=OPENAI_EMBEDDING_MODEL
        # )
        # .data[0]
        # .embedding,
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

    relevant_docs = [hit.payload for hit in hits]

    section_text_dict = defaultdict()

    for rd in relevant_docs:
        section_name = rd["sectionName"]
        if section_name not in section_text_dict:
            section_text_dict[section_name] = ""
        section_text_dict[section_name] += rd["text"] + ". "

    relevant_docs_sentences = ""

    for sec_name, section_text in section_text_dict.items():
        relevant_docs_sentences += sec_name + ": "
        relevant_docs_sentences += section_text
        relevant_docs_sentences += "\n\n"

    return relevant_docs_sentences
