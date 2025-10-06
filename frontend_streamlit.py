# frontend_streamlit.py
import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("Medical Appointment Chatbot (RAG Demo)")

if "session_id" not in st.session_state:
    r = requests.post(f"{API}/start_session")
    st.session_state.session_id = r.json()["session_id"]

lang = st.selectbox("Patient language", options=["en","hi","mr","ta","te"], index=0)
user_input = st.text_input("Type message (patient):")

if st.button("Send"):
    payload = {"session_id": st.session_state.session_id, "text": user_input, "lang": lang}
    r = requests.post(f"{API}/chat", json=payload)
    if r.status_code == 200:
        st.write("Bot:", r.json()["reply"])
    else:
        st.error("Error: " + r.text)

if st.button("Generate summary & report (doctor)"):
    r = requests.post(f"{API}/summary", data={"session_id": st.session_state.session_id, "patient_name":"DemoPatient"})
    if r.status_code == 200:
        res = r.json()
        st.write("Summary for doctor:")
        st.write(res["summary"])
        st.write("Report saved at:", res["report_file"])
    else:
        st.error("Error generating summary")
