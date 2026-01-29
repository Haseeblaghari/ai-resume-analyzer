import streamlit as st
from Resume_analyzer import app

st.set_page_config(
    page_title="Resume_analyzer"
)

st.title("Resume Analyzer")

resume_path = st.sidebar.file_uploader(
    "Upload You Resume",
    type=["pdf"]
)

jb_path = st.sidebar.file_uploader(
    "Upload Job Description",
    type=["pdf"]
)

if resume_path is not None and jb_path is not None:
    results = app.invoke({
        "resume_path": resume_path,
        "jb_path": jb_path
    })

    st.markdown("\n===== FINAL SCORE =====")
    st.markdown(results["final_score"])

    st.markdown("\n===== GAPS =====")
    st.markdown(results["gaps"])

    st.markdown("\n===== SUGGESTIONS =====")
    st.markdown(results["suggestions"])
else:
    st.info("Please upload both Resume and Job Description PDFs.")