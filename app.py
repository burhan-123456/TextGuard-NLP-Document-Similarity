
import streamlit as st
from nlp_utils import (
    preprocess_text,
    calculate_similarity,
    classify_similarity,
    read_txt_file,
    read_pdf_file,
    document_statistics,
    get_common_keywords,
    get_unique_keywords,
    split_sentences,
    find_duplicate_sentences,
    sentence_similarity_analysis,
    get_similar_sentences,
    plagiarism_risk_summary
)

st.set_page_config(
    page_title="TextGuard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 TextGuard")
st.subheader("NLP Document Similarity & Plagiarism Screening")

st.write(
    "Compare two documents using NLP, TF-IDF and cosine similarity."
)

st.divider()

method = st.radio(
    "Choose Input Method",
    ["Enter Text", "Upload Documents"],
    horizontal=True
)

text1, text2 = "", ""

if method == "Enter Text":
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

else:
    col1, col2 = st.columns(2)

    with col1:
        file1 = st.file_uploader(
            "Upload Document 1",
            type=["txt", "pdf"],
            key="file1"
        )

    with col2:
        file2 = st.file_uploader(
            "Upload Document 2",
            type=["txt", "pdf"],
            key="file2"
        )

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

    if text1:
        with st.expander("Preview Document 1"):
            st.text(text1[:3000])

    if text2:
        with st.expander("Preview Document 2"):
            st.text(text2[:3000])

st.divider()

st.subheader("NLP Settings")

col1, col2 = st.columns(2)

with col1:
    remove_stopwords = st.checkbox(
        "Remove Stopwords",
        value=False
    )

with col2:
    apply_stemming = st.checkbox(
        "Apply Stemming",
        value=False
    )

threshold = st.slider(
    "Sentence Similarity Threshold (%)",
    min_value=0,
    max_value=100,
    value=60
)

if st.button(
    "🔍 Compare Documents",
    type="primary",
    use_container_width=True
):
    if not text1.strip() or not text2.strip():
        st.warning("Please provide both documents.")

    else:
        score = calculate_similarity(
            text1,
            text2,
            remove_stopwords,
            apply_stemming
        )

        level = classify_similarity(score)

        st.divider()
        st.subheader("Similarity Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Similarity Score",
                f"{score:.2f}%"
            )

        with col2:
            st.metric(
                "Similarity Level",
                level
            )

        st.progress(
            min(max(score / 100, 0.0), 1.0)
        )

        if level == "High":
            st.error("High textual similarity detected.")
        elif level == "Moderate":
            st.warning("Moderate textual similarity detected.")
        else:
            st.success("Low textual similarity detected.")

        st.divider()

        st.subheader("Document Statistics")

        stats1 = document_statistics(text1)
        stats2 = document_statistics(text2)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Document 1")
            st.metric("Words", stats1["words"])
            st.metric("Characters", stats1["characters"])
            st.metric("Unique Words", stats1["unique_words"])
            st.metric("Sentences", stats1["sentences"])

        with col2:
            st.markdown("### Document 2")
            st.metric("Words", stats2["words"])
            st.metric("Characters", stats2["characters"])
            st.metric("Unique Words", stats2["unique_words"])
            st.metric("Sentences", stats2["sentences"])

        st.divider()

        st.subheader("Common Keywords")

        common = get_common_keywords(text1, text2)

        st.metric("Matching Keywords", len(common))
        st.write(", ".join(common) if common else "None")

        st.subheader("Unique Keywords")

        unique = get_unique_keywords(text1, text2)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Unique to Document 1**")
            st.write(", ".join(unique["document1"]) or "None")

        with col2:
            st.write("**Unique to Document 2**")
            st.write(", ".join(unique["document2"]) or "None")

        st.divider()

        st.subheader("Sentence Analysis")

        sentences1 = split_sentences(text1)
        sentences2 = split_sentences(text2)

        col1, col2 = st.columns(2)

        col1.metric("Document 1 Sentences", len(sentences1))
        col2.metric("Document 2 Sentences", len(sentences2))

        with st.expander("View Extracted Sentences"):
            st.write("**Document 1**")
            for i, sentence in enumerate(sentences1, 1):
                st.write(f"{i}. {sentence}")

            st.write("**Document 2**")
            for i, sentence in enumerate(sentences2, 1):
                st.write(f"{i}. {sentence}")

        st.divider()

        st.subheader("Exact Duplicate Sentences")

        duplicates = find_duplicate_sentences(text1, text2)

        if duplicates:
            st.warning(
                f"{len(duplicates)} duplicate sentence(s) found."
            )

            for sentence in duplicates:
                st.write(f"🔴 {sentence}")
        else:
            st.success("No exact duplicate sentences found.")

        st.divider()

        st.subheader("Sentence-Level Similarity")

        sentence_results = sentence_similarity_analysis(
            text1, text2
        )

        if sentence_results:
            st.dataframe(
                sentence_results,
                use_container_width=True
            )
        else:
            st.info("No sentences available for comparison.")

        st.divider()

        st.subheader("Potentially Similar Passages")

        similar = get_similar_sentences(
            text1, text2, threshold
        )

        if similar:
            for result in similar:
                st.markdown(
                    f"**Document 1:** {result['sentence']}"
                )

                st.markdown(
                    f"**Closest Match:** {result['best_match']}"
                )

                st.warning(
                    f"Similarity: {result['similarity']:.2f}%"
                )
        else:
            st.success(
                "No sentences exceed the selected threshold."
            )

        st.divider()

        st.subheader("Potential Plagiarism Risk Summary")

        summary = plagiarism_risk_summary(text1, text2)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Sentences Analyzed",
                summary["total_sentences"]
            )

        with col2:
            st.metric(
                "Flagged Sentences",
                summary["flagged_sentences"]
            )

        with col3:
            st.metric(
                "Flagged Percentage",
                f"{summary['flagged_percentage']:.2f}%"
            )

        risk = summary["risk"]

        st.write(f"**Similarity Risk Level: {risk}**")

        if risk == "High":
            st.error(
                "Many sentences are similar. Manual review is recommended."
            )
        elif risk == "Moderate":
            st.warning(
                "Some sentences may contain matching content."
            )
        elif risk == "Low":
            st.success(
                "Few sentences exceeded the similarity threshold."
            )
        else:
            st.info("Not enough text was available for analysis.")

        with st.expander("View Preprocessed Text"):
            st.write("**Document 1**")
            st.write(
                preprocess_text(
                    text1, remove_stopwords, apply_stemming
                )
            )

            st.write("**Document 2**")
            st.write(
                preprocess_text(
                    text2, remove_stopwords, apply_stemming
                )
            )

        st.info(
            "Textual similarity is not proof of plagiarism. "
            "Results require manual review."
        )
