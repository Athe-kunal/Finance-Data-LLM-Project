from src.chat_sec import get_openai_answer_sec
from src.queryDatabase import query_database_sec
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
sec_form_names = st.session_state["sec_form_names"]
form_name = st.selectbox("Form Name", tuple(sec_form_names))
st.session_state["form_name"] = form_name
ticker = st.session_state["ticker"]
year = st.session_state["year"]

st.title(f"{ticker}-{year}")


def generate_response(input_text):
    form = st.session_state["form_name"]
    relevant_text = query_database_sec(
        input_text, qdrant_client, encoder, search_form=form
    )
    res = get_openai_answer_sec(input_text, relevant_text)

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
    message = {"role": "assistant", "content": docs}
    st.session_state.messages.append(message)
