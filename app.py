import streamlit as st
from nlp_utils import calculate_similarity, preprocess_text

st.set_page_config(
    page_title="TextGuard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 TextGuard")
st.subheader("NLP Document Similarity Detector")

st.write("Analyze document similarity using TF-IDF and cosine similarity.")

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("Document 1", height=250)

with col2:
    text2 = st.text_area("Document 2", height=250)

if st.button("Compare Documents", type="primary"):
    if text1.strip() and text2.strip():
        score = calculate_similarity(text1, text2)

        st.divider()
        st.subheader("Similarity Analysis")

        st.metric(
            label="Document Similarity",
            value=f"{score:.2f}%"
        )

        st.progress(min(max(score / 100, 0.0), 1.0))

        with st.expander("View Preprocessed Text"):
            st.write("Document 1:")
            st.write(preprocess_text(text1))

            st.write("Document 2:")
            st.write(preprocess_text(text2))
    else:
        st.warning("Please enter both documents.")