
import streamlit as st
import pandas as pd
# pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt

from nlp_utils import (
    preprocess_text, calculate_similarity, jaccard_similarity,
    classify_similarity, read_txt_file, read_pdf_file,
    document_statistics, get_common_keywords, get_unique_keywords,
    split_sentences, find_duplicate_sentences, sentence_similarity_analysis,
    get_similar_sentences, plagiarism_risk_summary, similarity_distribution,
    statistics_comparison, sentence_similarity_matrix, sentence_results_csv,
    generate_text_report, word_frequency, top_keyword_frequencies,
    similarity_coverage, generate_json_report, analysis_summary
)

st.set_page_config(page_title="TextGuard", page_icon="🔍", layout="wide")
st.title("🔍 TextGuard")
st.subheader("NLP Document Similarity & Plagiarism Screening")
st.write("Compare documents using TF-IDF, cosine similarity, and Jaccard similarity.")

method = st.radio("Choose Input Method", ["Enter Text", "Upload Documents"], horizontal=True)
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
        file1 = st.file_uploader("Upload Document 1", type=["txt", "pdf"], key="f1")
    with c2:
        file2 = st.file_uploader("Upload Document 2", type=["txt", "pdf"], key="f2")

    def extract_file(file):
        if file is None:
            return ""
        try:
            return read_pdf_file(file) if file.name.lower().endswith(".pdf") else read_txt_file(file)
        except Exception as error:
            st.error(f"Unable to read {file.name}: {error}")
            return ""

    text1, text2 = extract_file(file1), extract_file(file2)
    if text1:
        with st.expander("Preview Document 1"):
            st.text(text1[:3000])
    if text2:
        with st.expander("Preview Document 2"):
            st.text(text2[:3000])

st.divider()
st.subheader("NLP Settings")
c1, c2 = st.columns(2)
with c1:
    remove_stopwords = st.checkbox("Remove Stopwords")
with c2:
    apply_stemming = st.checkbox("Apply Stemming")
threshold = st.slider("Sentence Similarity Threshold (%)", 0, 100, 60)

if st.button("🔍 Compare Documents", type="primary", use_container_width=True):
    if not text1.strip() or not text2.strip():
        st.warning("Please provide both documents.")
    else:
        score = calculate_similarity(text1, text2, remove_stopwords, apply_stemming)
        jaccard = jaccard_similarity(text1, text2, remove_stopwords, apply_stemming)
        results = sentence_similarity_analysis(text1, text2)
        summary = analysis_summary(text1, text2, score, threshold)
        stats = statistics_comparison(text1, text2)
        sentences1, sentences2 = split_sentences(text1), split_sentences(text2)
        coverage = similarity_coverage(text1, text2, threshold)

        st.divider()
        st.subheader("Analysis Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TF-IDF Similarity", f"{score:.2f}%")
        c2.metric("Jaccard Similarity", f"{jaccard:.2f}%")
        c3.metric("Matching Keywords", summary["common_keywords"])
        c4.metric("Exact Duplicates", summary["exact_duplicates"])
        st.progress(min(max(score / 100, 0.0), 1.0))
        st.write("**Similarity Level:**", summary["similarity_level"])

        if summary["similarity_level"] == "High":
            st.error("High textual similarity detected.")
        elif summary["similarity_level"] == "Moderate":
            st.warning("Moderate textual similarity detected.")
        else:
            st.success("Low textual similarity detected.")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview", "Sentence Analysis", "Visualizations", "Keywords", "Reports"
        ])

        with tab1:
            st.subheader("Document Statistics")
            c1, c2 = st.columns(2)
            c1.write("**Document 1**")
            c1.write(stats["Document 1"])
            c2.write("**Document 2**")
            c2.write(stats["Document 2"])
            st.subheader("Statistics Comparison Chart")
            st.bar_chart(pd.DataFrame(stats))

            st.subheader("Similarity Method Comparison")
            st.bar_chart(pd.DataFrame({
                "Method": ["TF-IDF Cosine", "Jaccard"],
                "Similarity (%)": [score, jaccard]
            }).set_index("Method"))

            st.subheader("Potential Plagiarism Risk Summary")
            risk = plagiarism_risk_summary(text1, text2)
            c1, c2, c3 = st.columns(3)
            c1.metric("Sentences Analyzed", risk["total_sentences"])
            c2.metric("Flagged Sentences", risk["flagged_sentences"])
            c3.metric("Flagged Percentage", f"{risk['flagged_percentage']:.2f}%")
            st.write("**Risk Level:**", risk["risk"])

            st.subheader("Bidirectional Similarity Coverage")
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

            st.subheader("Final Analysis Summary")
            st.write(
                f"TF-IDF similarity is **{score:.2f}%**, Jaccard similarity is "
                f"**{jaccard:.2f}%**, and **{summary['similar_sentences']}** "
                f"Document 1 sentences meet the {threshold}% threshold."
            )

        with tab2:
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
                file_name="sentence_similarity.csv", mime="text/csv"
            )

            st.subheader("Potentially Similar Passages")
            similar = [r for r in results if r["similarity"] >= threshold]
            if similar:
                for r in similar:
                    st.write("**Document 1:**", r["sentence"])
                    st.write("**Closest Match:**", r["best_match"])
                    st.warning(f"Similarity: {r['similarity']:.2f}%")
            else:
                st.success("No sentences exceed the selected threshold.")

        with tab3:
            st.subheader("Similarity Distribution")
            distribution = similarity_distribution(results)
            st.bar_chart(pd.DataFrame({
                "Level": list(distribution),
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
            else:
                st.info("No heatmap data available.")

            st.subheader("Top Keyword Frequency Charts")
            c1, c2 = st.columns(2)
            for column, text, label in [
                (c1, text1, "Document 1"), (c2, text2, "Document 2")
            ]:
                with column:
                    st.write(f"**{label}**")
                    keywords = top_keyword_frequencies(text)
                    if keywords:
                        st.bar_chart(pd.DataFrame({
                            "Keyword": list(keywords),
                            "Frequency": list(keywords.values())
                        }).set_index("Keyword"))
                    else:
                        st.info("No keywords available.")

        with tab4:
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

            st.subheader("Word Frequency Analysis")
            c1, c2 = st.columns(2)
            c1.write("**Document 1**")
            c1.dataframe(pd.DataFrame(list(word_frequency(text1).items()),
                                      columns=["Word", "Frequency"]))
            c2.write("**Document 2**")
            c2.dataframe(pd.DataFrame(list(word_frequency(text2).items()),
                                      columns=["Word", "Frequency"]))

        with tab5:
            st.subheader("Download Comparison Reports")
            txt_report = generate_text_report(text1, text2, score, results, threshold)
            json_report = generate_json_report(
                text1, text2, score, results, threshold,
                remove_stopwords, apply_stemming
            )
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Download TXT Report", txt_report.encode("utf-8"),
                                   "textguard_report.txt", "text/plain")
            with c2:
                st.download_button("Download JSON Report", json_report.encode("utf-8"),
                                   "textguard_report.json", "application/json")

            with st.expander("Preprocessed Text"):
                st.write("Document 1:", preprocess_text(text1, remove_stopwords, apply_stemming))
                st.write("Document 2:", preprocess_text(text2, remove_stopwords, apply_stemming))

        st.caption("Text similarity is not proof of plagiarism.")
