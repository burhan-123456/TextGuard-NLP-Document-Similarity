import streamlit as st
from nlp_utils import calculate_similarity, preprocess_text

st.set_page_config(
    page_title="TextGuard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 TextGuard")
st.subheader("NLP Document Similarity Detector")

st.write("Compare documents using cleaned and normalized text.")

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("Document 1", height=250)

with col2:
    text2 = st.text_area("Document 2", height=250)

if st.button("Compare Documents"):
    if text1.strip() and text2.strip():
        cleaned1 = preprocess_text(text1)
        cleaned2 = preprocess_text(text2)

        score = calculate_similarity(text1, text2)

        st.success(f"Similarity Score: {score:.2f}%")

        with st.expander("View Preprocessed Text"):
            st.write("Document 1:")
            st.write(cleaned1)

            st.write("Document 2:")
            st.write(cleaned2)
    else:
        st.warning("Please enter both documents.")