
import streamlit as st
import pandas as pd
# pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt

from nlp_utils import (
    preprocess_text, calculate_similarity, classify_similarity,
    read_txt_file, read_pdf_file, document_statistics,
    get_common_keywords, get_unique_keywords, split_sentences,
    find_duplicate_sentences, sentence_similarity_analysis,
    get_similar_sentences, plagiarism_risk_summary,
    similarity_distribution, statistics_comparison,
    sentence_similarity_matrix, sentence_results_csv,
    generate_text_report, word_frequency, top_keyword_frequencies,
    similarity_coverage, generate_json_report
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
        except Exception as error:
            st.error(f"File error: {error}")
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
        comparison = statistics_comparison(text1, text2)
        sentences1, sentences2 = split_sentences(text1), split_sentences(text2)

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
        c1.write("**Document 1**")
        c1.write(comparison["Document 1"])
        c2.write("**Document 2**")
        c2.write(comparison["Document 2"])

        st.subheader("Statistics Comparison Chart")
        st.bar_chart(pd.DataFrame(comparison))

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
        c1.metric("Document 1 Sentences", len(sentences1))
        c2.metric("Document 2 Sentences", len(sentences2))
        with st.expander("View Extracted Sentences"):
            st.write("Document 1:", sentences1)
            st.write("Document 2:", sentences2)

        st.subheader("Exact Duplicate Sentences")
        duplicates = find_duplicate_sentences(text1, text2)
        if duplicates:
            for sentence in duplicates:
                st.warning(sentence)
        else:
            st.success("No exact duplicate sentences found.")

        st.subheader("Sentence-Level Similarity")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        st.download_button(
            "Download Sentence Similarity CSV",
            data=sentence_results_csv(results),
            file_name="sentence_similarity.csv",
            mime="text/csv"
        )

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

        st.subheader("Sentence Similarity Heatmap")
        matrix = sentence_similarity_matrix(text1, text2)
        if matrix and sentences2:
            fig, ax = plt.subplots(figsize=(9, 5))
            image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
            ax.set_xlabel("Document 2 Sentences")
            ax.set_ylabel("Document 1 Sentences")
            ax.set_xticks(range(len(sentences2)))
            ax.set_yticks(range(len(sentences1)))
            ax.set_xticklabels(range(1, len(sentences2) + 1))
            ax.set_yticklabels(range(1, len(sentences1) + 1))
            fig.colorbar(image, ax=ax, label="Similarity (%)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Word Frequency Analysis")
        c1, c2 = st.columns(2)
        c1.write("**Document 1 - Top Words**")
        c1.dataframe(pd.DataFrame(list(word_frequency(text1).items()),
                                  columns=["Word", "Frequency"]))
        c2.write("**Document 2 - Top Words**")
        c2.dataframe(pd.DataFrame(list(word_frequency(text2).items()),
                                  columns=["Word", "Frequency"]))

        st.subheader("Top Keyword Frequency Charts")
        c1, c2 = st.columns(2)
        for column, text, label in [
            (c1, text1, "Document 1 Keywords"),
            (c2, text2, "Document 2 Keywords")
        ]:
            with column:
                st.write(f"**{label}**")
                keywords = top_keyword_frequencies(text)
                if keywords:
                    st.bar_chart(pd.DataFrame({
                        "Keyword": list(keywords.keys()),
                        "Frequency": list(keywords.values())
                    }).set_index("Keyword"))
                else:
                    st.info("No keywords available.")

        st.subheader("Bidirectional Similarity Coverage")
        coverage = similarity_coverage(text1, text2, threshold)
        c1, c2 = st.columns(2)
        c1.metric("Document 1 Coverage", f"{coverage['document1']:.2f}%")
        c2.metric("Document 2 Coverage", f"{coverage['document2']:.2f}%")
        st.bar_chart(pd.DataFrame({
            "Document": ["Document 1", "Document 2"],
            "Coverage (%)": [coverage["document1"], coverage["document2"]]
        }).set_index("Document"))
        st.caption(
            f"Document 1: {coverage['matched1']}/{coverage['total1']} matched. "
            f"Document 2: {coverage['matched2']}/{coverage['total2']} matched."
        )

        st.subheader("Download Comparison Reports")
        report = generate_text_report(text1, text2, score, results, threshold)
        json_report = generate_json_report(
            text1, text2, score, results, threshold,
            remove_stopwords, apply_stemming
        )

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download TXT Report",
                data=report.encode("utf-8"),
                file_name="textguard_report.txt",
                mime="text/plain"
            )
        with c2:
            st.download_button(
                "Download JSON Report",
                data=json_report.encode("utf-8"),
                file_name="textguard_report.json",
                mime="application/json"
            )

        with st.expander("Preprocessed Text"):
            st.write("Document 1:", preprocess_text(text1, remove_stopwords, apply_stemming))
            st.write("Document 2:", preprocess_text(text2, remove_stopwords, apply_stemming))

        st.caption("Similarity does not establish plagiarism.")
