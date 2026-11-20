import streamlit as st
from nlp_utils import (
    calculate_similarity,
    preprocess_text,
    classify_similarity
)

st.set_page_config(
    page_title="TextGuard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 TextGuard")
st.subheader("NLP Document Similarity Detector")

st.write("Compare documents and identify their similarity level.")

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("Document 1", height=250)

with col2:
    text2 = st.text_area("Document 2", height=250)

if st.button("Compare Documents", type="primary"):
    if text1.strip() and text2.strip():
        score = calculate_similarity(text1, text2)
        category = classify_similarity(score)

        st.divider()
        st.subheader("Similarity Analysis")

        col3, col4 = st.columns(2)

        with col3:
            st.metric("Similarity Score", f"{score:.2f}%")

        with col4:
            st.metric("Similarity Level", category)

        st.progress(min(max(score / 100, 0.0), 1.0))

        if category == "High":
            st.error("High textual similarity detected.")
        elif category == "Moderate":
            st.warning("Moderate textual similarity detected.")
        else:
            st.success("Low textual similarity detected.")

        with st.expander("View Preprocessed Text"):
            st.write("Document 1:")
            st.write(preprocess_text(text1))

            st.write("Document 2:")
            st.write(preprocess_text(text2))
    else:
        st.warning("Please enter both documents.")