# pyrefly: ignore[missing-import]

import streamlit as st
from nlp_utils import (
    calculate_similarity,
    classify_similarity,
    preprocess_text,
    read_txt_file,
    read_pdf_file,
    document_statistics
)

st.set_page_config(page_title="TextGuard", page_icon="🔍", layout="wide")

st.title("🔍 TextGuard")
st.subheader("NLP Document Similarity Detector")

method = st.radio(
    "Choose Input Method",
    ["Enter Text", "Upload Documents"],
    horizontal=True
)

text1, text2 = "", ""

if method == "Enter Text":
    col1, col2 = st.columns(2)

    with col1:
        text1 = st.text_area("Document 1", height=250)

    with col2:
        text2 = st.text_area("Document 2", height=250)

else:
    col1, col2 = st.columns(2)

    with col1:
        file1 = st.file_uploader("Document 1", type=["txt", "pdf"], key="f1")

    with col2:
        file2 = st.file_uploader("Document 2", type=["txt", "pdf"], key="f2")

    def extract_file(file):
        if file is None:
            return ""
        try:
            if file.name.lower().endswith(".pdf"):
                return read_pdf_file(file)
            return read_txt_file(file)
        except Exception as error:
            st.error(f"Unable to read {file.name}: {error}")
            return ""

    text1 = extract_file(file1)
    text2 = extract_file(file2)

if st.button("Compare Documents", type="primary"):
    if text1.strip() and text2.strip():
        score = calculate_similarity(text1, text2)
        level = classify_similarity(score)

        st.subheader("Similarity Results")
        col1, col2 = st.columns(2)
        col1.metric("Similarity", f"{score:.2f}%")
        col2.metric("Level", level)
        st.progress(min(max(score / 100, 0.0), 1.0))

        stats1 = document_statistics(text1)
        stats2 = document_statistics(text2)

        st.subheader("Document Statistics")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Document 1**")
            st.write(stats1)

        with col2:
            st.write("**Document 2**")
            st.write(stats2)

        with st.expander("Preprocessed Text"):
            st.write("Document 1:", preprocess_text(text1))
            st.write("Document 2:", preprocess_text(text2))
    else:
        st.warning("Please provide both documents.")