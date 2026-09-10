"""
Streamlit frontend — build this LAST, after the pipeline works via the API.

Run with: streamlit run app.py
(make sure the FastAPI backend is running at localhost:8000 first)
"""

import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AI Medical Scribe", layout="wide")
st.title("AI Medical Scribe")
st.caption("Audio → transcript → diarized → structured, grounded SOAP note")

tab1, tab2 = st.tabs(["Upload audio", "Paste transcript directly"])

with tab1:
    st.write("Not yet wired up — connect this to /transcribe and /diarize once "
             "those routes work end-to-end.")
    uploaded = st.file_uploader("Upload a conversation recording", type=["wav", "mp3", "m4a"])
    if uploaded and st.button("Process audio"):
        st.info("TODO: call /transcribe, then /diarize, then merge, then /generate-note")

with tab2:
    st.write("Use this tab first — it lets you test note generation without audio.")
    default_transcript = (
        "[DOCTOR]: What brings you in today?\n"
        "[PATIENT]: I've had a headache for two days, worse in the mornings.\n"
    )
    transcript = st.text_area("Labeled transcript", value=default_transcript, height=200)

    if st.button("Generate SOAP note"):
        with st.spinner("Generating..."):
            resp = requests.post(
                f"{API_BASE}/generate-note", params={"transcript": transcript}
            )
        if resp.ok:
            note = resp.json()
            for section in ["subjective", "objective", "assessment", "plan"]:
                st.subheader(section.capitalize())
                for field in note.get(section, []):
                    flag = "🔴" if field["confidence"] == "low" else (
                        "🟡" if field["confidence"] == "medium" else "🟢"
                    )
                    st.markdown(f"{flag} **{field['value']}**")
                    if field.get("source_span"):
                        st.caption(f"source: \"{field['source_span']}\"")
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")
