import streamlit as st
from nlp_utils import (
    calculate_similarity,
    preprocess_text
)

st.set_page_config(
    page_title="TextGuard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 TextGuard")
st.subheader("NLP Document Similarity Detector")

st.write(
    "Compare documents using TF-IDF vectorization "
    "and cosine similarity."
)

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("Document 1", height=250)

with col2:
    text2 = st.text_area("Document 2", height=250)

if st.button("Compare Documents"):
    if text1.strip() and text2.strip():
        score = calculate_similarity(text1, text2)

        st.success(f"TF-IDF Similarity: {score:.2f}%")

        with st.expander("View Preprocessed Text"):
            st.write("Document 1:")
            st.write(preprocess_text(text1))

            st.write("Document 2:")
            st.write(preprocess_text(text2))
    else:
        st.warning("Please enter both documents.")