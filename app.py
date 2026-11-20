import streamlit as st

st.set_page_config(
    page_title="TextGuard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 TextGuard")
st.subheader("NLP Document Similarity Detector")

st.write(
    "Welcome to TextGuard! "
    "Compare documents and identify text similarities "
    "using Natural Language Processing."
)

st.info("Start by entering text to compare.")