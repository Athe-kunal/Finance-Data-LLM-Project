import streamlit as st
from src.vectorDatabaseDocker import create_database
from datetime import datetime

curr_year = datetime.now().year
ticker = st.text_input(label="Ticker")
year = st.text_input(label="Year")

if year != "":
    int_year = int(float(year))
submit_button = st.button(label="Submit")
if ticker != "" and year != "" and submit_button:
    if curr_year == int_year:
        curr_year_bool = True
    else:
        curr_year_bool = False
    (
        qdrant_client,
        encoder,
        speakers_list_1,
        speakers_list_2,
        speakers_list_3,
        speakers_list_4,
        sec_form_names,
        earnings_call_quarter_vals,
    ) = create_database(ticker=ticker, year=int_year)
    st.write("Created the database")

    st.session_state["ticker"] = ticker
    st.session_state["year"] = str(year)
    st.session_state["qdrant_client"] = qdrant_client
    st.session_state["encoder"] = encoder
    st.session_state["speaker_list_1"] = speakers_list_1
    st.session_state["speaker_list_2"] = speakers_list_2
    st.session_state["speaker_list_3"] = speakers_list_3
    st.session_state["speaker_list_4"] = speakers_list_4
    st.session_state["sec_form_names"] = sec_form_names
    st.session_state["earnings_call_quarter_vals"] = earnings_call_quarter_vals
