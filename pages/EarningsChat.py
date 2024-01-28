from src.chat_earnings_call import get_openai_answer_earnings_call
from src.vectorDatabase import query_database_earnings_call

import streamlit as st
from dotenv import load_dotenv
import openai
import os
import re

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

qdrant_client = st.session_state['qdrant_client']
encoder = st.session_state['encoder']

speaker_list_1 = st.session_state['speaker_list_1']
speakers_list_2 = st.session_state['speaker_list_2']
speakers_list_3 = st.session_state['speaker_list_3']

def generate_response(input_text,quarters:str):

    if quarters == "Q1":
        speakers_list = speaker_list_1
    elif quarters == "Q2":
        speakers_list = speakers_list_2
    elif quarters == "Q3":
        speakers_list = speakers_list_3

    relevant_text = query_database_earnings_call(input_text,qdrant_client,encoder,speakers_list)
    res = get_openai_answer_earnings_call(input_text,relevant_text)

    return res, relevant_text

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input():
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Answering..."):
            docs,relevant_text = generate_response(prompt) 
            docs = re.sub(r'\$', r'\\$',docs)
            relevant_text = re.sub(r'\$', r'\\$',relevant_text)
            st.write(docs)
            expander = st.expander("See relevant sources")
            expander.write(relevant_text)
    message = {"role": "assistant", "content": docs}
    st.session_state.messages.append(message)