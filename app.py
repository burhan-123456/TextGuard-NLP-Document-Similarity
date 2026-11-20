
import streamlit as st
import pandas as pd
from nlp_utils import (
    preprocess_text, calculate_similarity, classify_similarity,
    read_txt_file, read_pdf_file, document_statistics,
    get_common_keywords, get_unique_keywords, split_sentences,
    find_duplicate_sentences, sentence_similarity_analysis,
    get_similar_sentences, plagiarism_risk_summary, similarity_distribution
)

st.set_page_config(page_title="TextGuard", page_icon="🔍", layout="wide")
st.title("🔍 TextGuard")
st.subheader("NLP Document Similarity & Plagiarism Screening")

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

    text1, text2 = extract_file(file1), extract_file(file2)

st.subheader("NLP Settings")
remove_stopwords = st.checkbox("Remove Stopwords")
apply_stemming = st.checkbox("Apply Stemming")
threshold = st.slider("Sentence Similarity Threshold (%)", 0, 100, 60)

if st.button("Compare Documents", type="primary"):
    if not text1.strip() or not text2.strip():
        st.warning("Please provide both documents.")
    else:
        score = calculate_similarity(text1, text2, remove_stopwords, apply_stemming)
        level = classify_similarity(score)
        results = sentence_similarity_analysis(text1, text2)

        st.subheader("Similarity Results")
        c1, c2 = st.columns(2)
        c1.metric("Similarity Score", f"{score:.2f}%")
        c2.metric("Similarity Level", level)
        st.progress(min(max(score / 100, 0.0), 1.0))

        if level == "High":
            st.error("High textual similarity detected.")
        elif level == "Moderate":
            st.warning("Moderate textual similarity detected.")
        else:
            st.success("Low textual similarity detected.")

        st.subheader("Document Statistics")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Document 1**")
            st.write(document_statistics(text1))
        with c2:
            st.write("**Document 2**")
            st.write(document_statistics(text2))

        st.subheader("Common Keywords")
        common = get_common_keywords(text1, text2)
        st.metric("Matching Keywords", len(common))
        st.write(", ".join(common) or "None")

        st.subheader("Unique Keywords")
        unique = get_unique_keywords(text1, text2)
        c1, c2 = st.columns(2)
        c1.write("**Document 1 Only**")
        c1.write(", ".join(unique["document1"]) or "None")
        c2.write("**Document 2 Only**")
        c2.write(", ".join(unique["document2"]) or "None")

        st.subheader("Sentence Analysis")
        c1, c2 = st.columns(2)
        c1.metric("Document 1 Sentences", len(split_sentences(text1)))
        c2.metric("Document 2 Sentences", len(split_sentences(text2)))

        with st.expander("View Extracted Sentences"):
            st.write("Document 1:", split_sentences(text1))
            st.write("Document 2:", split_sentences(text2))

        st.subheader("Exact Duplicate Sentences")
        duplicates = find_duplicate_sentences(text1, text2)
        if duplicates:
            for sentence in duplicates:
                st.warning(sentence)
        else:
            st.success("No exact duplicate sentences found.")

        st.subheader("Sentence-Level Similarity")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.subheader("Potentially Similar Passages")
        similar = [r for r in results if r["similarity"] >= threshold]
        if similar:
            for result in similar:
                st.write("**Document 1:**", result["sentence"])
                st.write("**Closest Match:**", result["best_match"])
                st.warning(f"Similarity: {result['similarity']:.2f}%")
        else:
            st.success("No sentences exceed the selected threshold.")

        st.subheader("Potential Plagiarism Risk Summary")
        summary = plagiarism_risk_summary(text1, text2)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentences Analyzed", summary["total_sentences"])
        c2.metric("Flagged Sentences", summary["flagged_sentences"])
        c3.metric("Flagged Percentage", f"{summary['flagged_percentage']:.2f}%")
        st.write("**Similarity Risk Level:**", summary["risk"])

        st.subheader("Similarity Distribution")
        distribution = similarity_distribution(results)
        st.bar_chart(pd.DataFrame({
            "Level": list(distribution.keys()),
            "Sentences": list(distribution.values())
        }).set_index("Level"))

        with st.expander("Preprocessed Text"):
            st.write("Document 1:", preprocess_text(text1, remove_stopwords, apply_stemming))
            st.write("Document 2:", preprocess_text(text2, remove_stopwords, apply_stemming))

        st.caption("Similarity scores are screening indicators, not proof of plagiarism.")
