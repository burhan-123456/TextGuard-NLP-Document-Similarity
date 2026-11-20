
import streamlit as st
from nlp_utils import (
    preprocess_text, calculate_similarity, classify_similarity,
    read_txt_file, read_pdf_file, document_statistics,
    get_common_keywords, get_unique_keywords
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

st.subheader("NLP Settings")
remove_stopwords = st.checkbox("Remove Stopwords")
apply_stemming = st.checkbox("Apply Stemming")

if st.button("Compare Documents", type="primary"):
    if text1.strip() and text2.strip():
        score = calculate_similarity(text1, text2, remove_stopwords, apply_stemming)

        st.subheader("Similarity Results")
        c1, c2 = st.columns(2)
        c1.metric("Similarity", f"{score:.2f}%")
        c2.metric("Level", classify_similarity(score))
        st.progress(min(max(score / 100, 0.0), 1.0))

        st.subheader("Document Statistics")
        c1, c2 = st.columns(2)
        c1.write(document_statistics(text1))
        c2.write(document_statistics(text2))

        st.subheader("Common Keywords")
        common = get_common_keywords(text1, text2)
        st.metric("Matching Keywords", len(common))
        st.write(", ".join(common) or "None")

        st.subheader("Unique Keywords")
        unique = get_unique_keywords(text1, text2)
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Document 1 Only**")
            st.write(", ".join(unique["document1"]) or "None")
        with c2:
            st.write("**Document 2 Only**")
            st.write(", ".join(unique["document2"]) or "None")

        with st.expander("Preprocessed Text"):
            st.write("Document 1:", preprocess_text(text1, remove_stopwords, apply_stemming))
            st.write("Document 2:", preprocess_text(text2, remove_stopwords, apply_stemming))
    else:
        st.warning("Please provide both documents.")
