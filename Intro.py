import streamlit as st
from src.vectorDatabase import create_database

ticker = st.text_input(label="Ticker")
year = st.text_input(label="Year")

if year != "":
    int_year = int(float(year))

if ticker != "" and year != "":
    (
        qdrant_client,
        encoder,
        speakers_list_1,
        speakers_list_2,
        speakers_list_3,
        sec_form_names,
        earnings_call_quarter_vals,
    ) = create_database(ticker=ticker, year=int_year)
    st.write("Created the database")

    st.session_state["qdrant_client"] = qdrant_client
    st.session_state["encoder"] = encoder
    st.session_state["speaker_list_1"] = speakers_list_1
    st.session_state["speaker_list_2"] = speakers_list_2
    st.session_state["speaker_list_3"] = speakers_list_3
    st.session_state["sec_form_names"] = sec_form_names
    st.session_state["earnings_call_quarter_vals"] = earnings_call_quarter_vals
