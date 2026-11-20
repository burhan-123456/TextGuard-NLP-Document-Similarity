
import streamlit as st
from nlp_utils import (
    preprocess_text, calculate_similarity, classify_similarity,
    read_txt_file, read_pdf_file, document_statistics
)

st.set_page_config(page_title="TextGuard", page_icon="🔍", layout="wide")
st.title("🔍 TextGuard")
st.subheader("NLP Document Similarity Detector")

method = st.radio("Input Method", ["Enter Text", "Upload Documents"], horizontal=True)
text1, text2 = "", ""

if method == "Enter Text":
    c1, c2 = st.columns(2)
    with c1:
        text1 = st.text_area("Document 1", height=250)
    with c2:
        text2 = st.text_area("Document 2", height=250)
else:
    c1, c2 = st.columns(2)
    with c1:
        file1 = st.file_uploader("Document 1", type=["txt", "pdf"], key="f1")
    with c2:
        file2 = st.file_uploader("Document 2", type=["txt", "pdf"], key="f2")

    def extract_file(file):
        if file is None:
            return ""
        try:
            return read_pdf_file(file) if file.name.lower().endswith(".pdf") else read_txt_file(file)
        except Exception as e:
            st.error(f"File error: {e}")
            return ""

    text1 = extract_file(file1)
    text2 = extract_file(file2)

if st.button("Compare Documents", type="primary"):
    if text1.strip() and text2.strip():
        score = calculate_similarity(text1, text2)
        level = classify_similarity(score)

        st.subheader("Similarity Results")
        c1, c2 = st.columns(2)
        c1.metric("Similarity", f"{score:.2f}%")
        c2.metric("Level", level)
        st.progress(min(max(score / 100, 0.0), 1.0))

        st.subheader("Document Statistics")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Document 1", document_statistics(text1))
        with c2:
            st.write("Document 2", document_statistics(text2))

        with st.expander("Preprocessed Text (Stopwords Removed)"):
            st.write("Document 1:", preprocess_text(text1, True))
            st.write("Document 2:", preprocess_text(text2, True))
    else:
        st.warning("Please provide both documents.")
