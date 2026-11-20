import streamlit as st

st.set_page_config(
    page_title="TextGuard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 TextGuard")
st.subheader("NLP Document Similarity Detector")

st.write(
    "Compare two text documents using Natural Language Processing."
)

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area(
        "Document 1",
        placeholder="Enter first document...",
        height=250
    )

with col2:
    text2 = st.text_area(
        "Document 2",
        placeholder="Enter second document...",
        height=250
    )

if st.button("Compare Documents"):
    if text1.strip() and text2.strip():
        st.success("Both documents are ready for comparison.")
    else:
        st.warning("Please enter text in both documents.")