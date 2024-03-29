from src.chat_earnings_call import get_openai_answer_earnings_call
from src.queryDatabase import query_database_earnings_call
from streamlit_feedback import streamlit_feedback
import streamlit as st
from dotenv import load_dotenv
import openai
import os
import re

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

qdrant_client = st.session_state["qdrant_client"]
encoder = st.session_state["encoder"]

speaker_list_1 = st.session_state["speaker_list_1"]
speakers_list_2 = st.session_state["speaker_list_2"]
speakers_list_3 = st.session_state["speaker_list_3"]
speakers_list_4 = st.session_state["speaker_list_4"]
earnings_call_quarter_vals = st.session_state["earnings_call_quarter_vals"]
quarter = st.selectbox("Quarter Name", tuple(earnings_call_quarter_vals))

st.session_state["quarter"] = quarter
ticker = st.session_state["ticker"]
year = st.session_state["year"]

st.title(f"{ticker}-{year}")


def generate_response(input_text):
    quarter = st.session_state["quarter"]
    if quarter == "Q1":
        speakers_list = speaker_list_1
    elif quarter == "Q2":
        speakers_list = speakers_list_2
    elif quarter == "Q3":
        speakers_list = speakers_list_3
    elif quarter == "Q4":
        speakers_list = speakers_list_4

    relevant_text = query_database_earnings_call(
        input_text, quarter, qdrant_client, encoder, speakers_list
    )
    res = get_openai_answer_earnings_call(input_text, relevant_text)

    return res, relevant_text


if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, how can I help you?"}
    ]

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
            docs, relevant_text = generate_response(prompt)
            docs = re.sub(r"\$", r"\\$", docs)
            relevant_text = re.sub(r"\$", r"\\$", relevant_text)
            st.write(docs)
            expander = st.expander("See relevant sources")
            expander.write(relevant_text)
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="Please describe the feedback in detail",
            )
            # print(feedback)
    message = {"role": "assistant", "content": docs}
    st.session_state.messages.append(message)
