
import io

import streamlit as st
import pandas as pd
# pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt

from database import (
    initialize_database,
    save_comparison,
    get_history,
    delete_comparison,
    clear_history,
    export_history_csv,
    export_history_json,
    history_analytics
)

from nlp_utils import (
    preprocess_text,
    calculate_similarity,
    jaccard_similarity,
    bigram_similarity,
    common_bigrams,
    longest_common_word_sequence,
    read_txt_file,
    read_pdf_file,
    get_common_keywords,
    get_unique_keywords,
    split_sentences,
    find_duplicate_sentences,
    sentence_similarity_analysis,
    plagiarism_risk_summary,
    similarity_distribution,
    statistics_comparison,
    sentence_similarity_matrix,
    sentence_results_csv,
    generate_text_report,
    word_frequency,
    top_keyword_frequencies,
    similarity_coverage,
    generate_json_report,
    analysis_summary,
    highlight_sentence,
    sort_sentence_results,
    search_sentence_results,
    generate_html_report,
    word_level_comparison,
    create_history_entry
)


st.set_page_config(
    page_title="TextGuard",
    page_icon="🔍",
    layout="wide"
)

initialize_database()

st.title("🔍 TextGuard")
st.subheader(
    "NLP Document Similarity & Plagiarism Screening"
)

st.write(
    "Compare two documents using NLP, analyze matching "
    "sentences, and explore saved comparison history."
)

st.divider()

# -------------------------------
# DOCUMENT INPUT
# -------------------------------

method = st.radio(
    "Input Method",
    ["Enter Text", "Upload Documents"],
    horizontal=True
)

text1 = ""
text2 = ""

if method == "Enter Text":

    c1, c2 = st.columns(2)

    with c1:
        text1 = st.text_area(
            "Document 1",
            height=250,
            placeholder="Enter first document..."
        )

    with c2:
        text2 = st.text_area(
            "Document 2",
            height=250,
            placeholder="Enter second document..."
        )

else:

    c1, c2 = st.columns(2)

    with c1:
        file1 = st.file_uploader(
            "Upload Document 1",
            type=["txt", "pdf"],
            key="f1"
        )

    with c2:
        file2 = st.file_uploader(
            "Upload Document 2",
            type=["txt", "pdf"],
            key="f2"
        )

    def extract_file(file):
        if file is None:
            return ""

        try:
            if file.name.lower().endswith(".pdf"):
                return read_pdf_file(file)

            return read_txt_file(file)

        except Exception as error:
            st.error(f"File error: {error}")
            return ""

    text1 = extract_file(file1)
    text2 = extract_file(file2)

    for number, text in [
        (1, text1),
        (2, text2)
    ]:
        if text:
            with st.expander(
                f"Preview Document {number}"
            ):
                st.text(text[:3000])


# -------------------------------
# NLP SETTINGS
# -------------------------------

st.divider()
st.subheader("NLP Settings")

c1, c2 = st.columns(2)

with c1:
    remove_stopwords = st.checkbox(
        "Remove Stopwords"
    )

with c2:
    apply_stemming = st.checkbox(
        "Apply Stemming"
    )

threshold = st.slider(
    "Sentence Similarity Threshold (%)",
    0, 100, 60
)

min_words = st.slider(
    "Minimum Words Per Sentence",
    1, 20, 1
)

keyword_limit = st.slider(
    "Number of Top Keywords",
    5, 30, 10
)


# -------------------------------
# DOCUMENT COMPARISON
# -------------------------------

if st.button(
    "🔍 Compare Documents",
    type="primary",
    use_container_width=True
):

    if not text1.strip() or not text2.strip():
        st.warning(
            "Please provide both documents."
        )

    else:

        score = calculate_similarity(
            text1,
            text2,
            remove_stopwords,
            apply_stemming
        )

        jaccard = jaccard_similarity(
            text1,
            text2,
            remove_stopwords,
            apply_stemming
        )

        bigram = bigram_similarity(
            text1,
            text2,
            remove_stopwords,
            apply_stemming
        )

        results = sentence_similarity_analysis(
            text1,
            text2,
            min_words
        )

        summary = analysis_summary(
            text1,
            text2,
            score,
            threshold,
            min_words
        )

        coverage = similarity_coverage(
            text1,
            text2,
            threshold,
            min_words
        )

        sentences1 = split_sentences(
            text1, min_words
        )

        sentences2 = split_sentences(
            text2, min_words
        )

        save_comparison(
            create_history_entry(
                text1,
                text2,
                score,
                threshold,
                min_words,
                remove_stopwords,
                apply_stemming
            )
        )

        st.divider()
        st.subheader("Analysis Dashboard")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "TF-IDF Similarity",
            f"{score:.2f}%"
        )

        c2.metric(
            "Jaccard Similarity",
            f"{jaccard:.2f}%"
        )

        c3.metric(
            "Bigram Similarity",
            f"{bigram:.2f}%"
        )

        c4.metric(
            "Exact Duplicates",
            summary["exact_duplicates"]
        )

        st.progress(
            min(max(score / 100, 0), 1.0)
        )

        if summary["similarity_level"] == "High":
            st.error(
                "High textual similarity detected."
            )

        elif summary["similarity_level"] == "Moderate":
            st.warning(
                "Moderate textual similarity detected."
            )

        else:
            st.success(
                "Low textual similarity detected."
            )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview",
            "Sentence Analysis",
            "Visualizations",
            "Keywords",
            "Reports"
        ])

        # -------------------------------
        # OVERVIEW
        # -------------------------------

        with tab1:

            st.subheader("Document Statistics")

            stats = statistics_comparison(
                text1, text2
            )

            c1, c2 = st.columns(2)

            with c1:
                st.write("**Document 1**")
                st.write(stats["Document 1"])

            with c2:
                st.write("**Document 2**")
                st.write(stats["Document 2"])

            st.bar_chart(
                pd.DataFrame(stats)
            )

            st.subheader("Similarity Methods")

            method_data = pd.DataFrame({
                "Method": [
                    "TF-IDF",
                    "Jaccard",
                    "Bigram"
                ],
                "Score": [
                    score,
                    jaccard,
                    bigram
                ]
            }).set_index("Method")

            st.bar_chart(method_data)

            st.subheader(
                "Longest Common Word Sequence"
            )

            longest = longest_common_word_sequence(
                text1, text2
            )

            st.metric(
                "Matching Words",
                longest["length"]
            )

            st.info(
                longest["sequence"] or "None"
            )

            st.subheader(
                "Potential Plagiarism Risk"
            )

            risk = plagiarism_risk_summary(
                text1,
                text2,
                70,
                min_words
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Sentences Analyzed",
                risk["total_sentences"]
            )

            c2.metric(
                "Flagged Sentences",
                risk["flagged_sentences"]
            )

            c3.metric(
                "Flagged Percentage",
                f"{risk['flagged_percentage']:.2f}%"
            )

            st.write(
                "**Risk Level:**",
                risk["risk"]
            )

            st.subheader(
                "Bidirectional Similarity Coverage"
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Document 1 Coverage",
                f"{coverage['document1']:.2f}%"
            )

            c2.metric(
                "Document 2 Coverage",
                f"{coverage['document2']:.2f}%"
            )

            coverage_data = pd.DataFrame({
                "Document": [
                    "Document 1",
                    "Document 2"
                ],
                "Coverage": [
                    coverage["document1"],
                    coverage["document2"]
                ]
            }).set_index("Document")

            st.bar_chart(coverage_data)

            st.caption(
                f"Matched: "
                f"{coverage['matched1']}/{coverage['total1']} "
                f"and "
                f"{coverage['matched2']}/{coverage['total2']}."
            )

            st.subheader(
                "Final Analysis Summary"
            )

            st.write(
                f"Overall similarity: **{score:.2f}%**. "
                f"Similar sentences: "
                f"**{summary['similar_sentences']}**."
            )

        # -------------------------------
        # SENTENCE ANALYSIS
        # -------------------------------

        with tab2:

            st.subheader("Sentence Analysis")

            c1, c2 = st.columns(2)

            c1.metric(
                "Document 1 Sentences",
                len(sentences1)
            )

            c2.metric(
                "Document 2 Sentences",
                len(sentences2)
            )

            with st.expander(
                "Extracted Sentences"
            ):
                st.write(
                    "**Document 1:**",
                    sentences1
                )

                st.write(
                    "**Document 2:**",
                    sentences2
                )

            st.subheader(
                "Exact Duplicate Sentences"
            )

            duplicates = find_duplicate_sentences(
                text1,
                text2,
                min_words
            )

            if duplicates:
                for sentence in duplicates:
                    st.warning(sentence)
            else:
                st.success(
                    "No exact duplicates found."
                )

            st.subheader(
                "Sentence-Level Similarity"
            )

            sort_order = st.selectbox(
                "Sort Sentence Matches",
                [
                    "Highest First",
                    "Lowest First"
                ]
            )

            search_query = st.text_input(
                "Search Sentence Matches"
            )

            filtered = search_sentence_results(
                sort_sentence_results(
                    results,
                    sort_order
                ),
                search_query
            )

            st.caption(
                f"Showing {len(filtered)} "
                f"of {len(results)} matches."
            )

            st.dataframe(
                pd.DataFrame(
                    filtered,
                    columns=[
                        "sentence",
                        "best_match",
                        "similarity"
                    ]
                ),
                use_container_width=True
            )

            st.download_button(
                "Download Sentence CSV",
                data=sentence_results_csv(
                    filtered
                ),
                file_name="sentence_similarity.csv",
                mime="text/csv"
            )

            st.subheader(
                "Potentially Similar Passages"
            )

            similar = [
                result
                for result in filtered
                if result["similarity"] >= threshold
            ]

            if similar:

                for result in similar:
                    st.write(
                        "**Document 1:**",
                        result["sentence"]
                    )

                    st.write(
                        "**Closest Match:**",
                        result["best_match"]
                    )

                    st.warning(
                        f"Similarity: "
                        f"{result['similarity']:.2f}%"
                    )

            else:
                st.info(
                    "No matches meet the threshold."
                )

            st.subheader(
                "Color-Coded Comparison"
            )

            for result in filtered:

                st.markdown(
                    highlight_sentence(
                        result["sentence"],
                        result["similarity"],
                        threshold
                    ),
                    unsafe_allow_html=True
                )

                if result["best_match"]:
                    st.write(
                        "**Closest Document 2 Match:**",
                        result["best_match"]
                    )

        # -------------------------------
        # VISUALIZATIONS
        # -------------------------------

        with tab3:

            st.subheader(
                "Similarity Distribution"
            )

            distribution = similarity_distribution(
                results
            )

            distribution_data = pd.DataFrame({
                "Level": list(distribution),
                "Sentences": list(
                    distribution.values()
                )
            }).set_index("Level")

            st.bar_chart(distribution_data)

            st.subheader(
                "Sentence Similarity Heatmap"
            )

            matrix = sentence_similarity_matrix(
                text1,
                text2,
                min_words
            )

            if matrix and sentences2:

                fig, ax = plt.subplots(
                    figsize=(9, 5)
                )

                image = ax.imshow(
                    matrix,
                    cmap="YlOrRd",
                    vmin=0,
                    vmax=100,
                    aspect="auto"
                )

                ax.set_xlabel(
                    "Document 2 Sentences"
                )

                ax.set_ylabel(
                    "Document 1 Sentences"
                )

                ax.set_xticks(
                    range(len(sentences2))
                )

                ax.set_yticks(
                    range(len(sentences1))
                )

                ax.set_xticklabels(
                    range(1, len(sentences2) + 1)
                )

                ax.set_yticklabels(
                    range(1, len(sentences1) + 1)
                )

                fig.colorbar(
                    image,
                    ax=ax,
                    label="Similarity (%)"
                )

                fig.tight_layout()

                st.pyplot(fig)

                buffer = io.BytesIO()

                fig.savefig(
                    buffer,
                    format="png",
                    dpi=160,
                    bbox_inches="tight"
                )

                st.download_button(
                    "Download Heatmap PNG",
                    data=buffer.getvalue(),
                    file_name="textguard_heatmap.png",
                    mime="image/png"
                )

                plt.close(fig)

            else:
                st.info(
                    "No heatmap data available."
                )

            st.subheader(
                f"Top {keyword_limit} Keyword Charts"
            )

            c1, c2 = st.columns(2)

            for column, text, label in [
                (c1, text1, "Document 1"),
                (c2, text2, "Document 2")
            ]:

                with column:
                    st.write(f"**{label}**")

                    keywords = top_keyword_frequencies(
                        text,
                        keyword_limit
                    )

                    if keywords:

                        keyword_data = pd.DataFrame({
                            "Keyword": list(keywords),
                            "Frequency": list(
                                keywords.values()
                            )
                        }).set_index("Keyword")

                        st.bar_chart(keyword_data)

                    else:
                        st.info(
                            "No keywords available."
                        )

        # -------------------------------
        # KEYWORDS
        # -------------------------------

        with tab4:

            st.subheader("Common Keywords")

            common = get_common_keywords(
                text1, text2
            )

            st.metric(
                "Matching Keywords",
                len(common)
            )

            st.write(
                ", ".join(common) or "None"
            )

            st.subheader("Unique Keywords")

            unique = get_unique_keywords(
                text1, text2
            )

            c1, c2 = st.columns(2)

            with c1:
                st.write(
                    "**Document 1 Only**"
                )
                st.write(
                    ", ".join(
                        unique["document1"]
                    ) or "None"
                )

            with c2:
                st.write(
                    "**Document 2 Only**"
                )
                st.write(
                    ", ".join(
                        unique["document2"]
                    ) or "None"
                )

            st.subheader("Common Bigrams")

            st.write(
                ", ".join(
                    common_bigrams(text1, text2)
                ) or "None"
            )

            st.subheader("Word Frequencies")

            c1, c2 = st.columns(2)

            with c1:
                st.write("**Document 1**")

                st.dataframe(
                    pd.DataFrame(
                        list(
                            word_frequency(
                                text1,
                                keyword_limit
                            ).items()
                        ),
                        columns=[
                            "Word",
                            "Frequency"
                        ]
                    ),
                    use_container_width=True
                )

            with c2:
                st.write("**Document 2**")

                st.dataframe(
                    pd.DataFrame(
                        list(
                            word_frequency(
                                text2,
                                keyword_limit
                            ).items()
                        ),
                        columns=[
                            "Word",
                            "Frequency"
                        ]
                    ),
                    use_container_width=True
                )

            st.subheader(
                "Word-Level Comparison"
            )

            word_rows = word_level_comparison(
                text1, text2
            )

            word_filter = st.selectbox(
                "Filter Words",
                [
                    "All",
                    "Common",
                    "Document 1 Only",
                    "Document 2 Only"
                ]
            )

            if word_filter != "All":
                word_rows = [
                    row
                    for row in word_rows
                    if row["Status"] == word_filter
                ]

            st.metric(
                "Displayed Words",
                len(word_rows)
            )

            st.dataframe(
                pd.DataFrame(
                    word_rows,
                    columns=[
                        "Word",
                        "Document 1 Frequency",
                        "Document 2 Frequency",
                        "Status"
                    ]
                ),
                use_container_width=True
            )

        # -------------------------------
        # REPORTS
        # -------------------------------

        with tab5:

            st.subheader("Download Reports")

            txt_report = generate_text_report(
                text1,
                text2,
                score,
                results,
                threshold,
                min_words
            )

            json_report = generate_json_report(
                text1,
                text2,
                score,
                results,
                threshold,
                remove_stopwords,
                apply_stemming,
                min_words
            )

            html_report = generate_html_report(
                text1,
                text2,
                score,
                results,
                threshold,
                min_words,
                remove_stopwords,
                apply_stemming
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.download_button(
                    "Download TXT",
                    data=txt_report.encode(
                        "utf-8"
                    ),
                    file_name="textguard_report.txt",
                    mime="text/plain"
                )

            with c2:
                st.download_button(
                    "Download JSON",
                    data=json_report.encode(
                        "utf-8"
                    ),
                    file_name="textguard_report.json",
                    mime="application/json"
                )

            with c3:
                st.download_button(
                    "Download HTML",
                    data=html_report.encode(
                        "utf-8"
                    ),
                    file_name="textguard_report.html",
                    mime="text/html"
                )

            with st.expander(
                "Preprocessed Text"
            ):

                st.write("**Document 1:**")
                st.write(
                    preprocess_text(
                        text1,
                        remove_stopwords,
                        apply_stemming
                    )
                )

                st.write("**Document 2:**")
                st.write(
                    preprocess_text(
                        text2,
                        remove_stopwords,
                        apply_stemming
                    )
                )

        st.caption(
            "Text similarity is not proof of plagiarism."
        )


# -------------------------------
# SAVED COMPARISON HISTORY
# -------------------------------

st.divider()
st.subheader("📜 Saved Comparison History")

search_id = st.text_input(
    "Search by Comparison ID"
)

c1, c2 = st.columns(2)

with c1:
    level_filter = st.selectbox(
        "Similarity Level",
        [
            "All",
            "High",
            "Moderate",
            "Low"
        ]
    )

with c2:
    risk_filter = st.selectbox(
        "Risk Level",
        [
            "All",
            "High",
            "Moderate",
            "Low",
            "Not Available"
        ]
    )

history = get_history(
    search_id,
    level_filter,
    risk_filter
)

if history:

    st.metric(
        "Matching Records",
        len(history)
    )

    st.dataframe(
        pd.DataFrame(history),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Delete a Comparison"
    )

    selected_id = st.selectbox(
        "Select Comparison ID",
        [
            record["id"]
            for record in history
        ]
    )

    if st.button(
        "Delete Selected Comparison"
    ):
        if delete_comparison(selected_id):
            st.success(
                f"Comparison {selected_id} deleted."
            )
            st.rerun()

else:
    st.info(
        "No matching records found."
    )


# -------------------------------
# EXPORT COMPLETE HISTORY
# -------------------------------

st.subheader(
    "Export Complete History"
)

all_history = get_history()

if all_history:

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            "Download All History CSV",
            data=export_history_csv(),
            file_name="textguard_all_history.csv",
            mime="text/csv"
        )

    with c2:
        st.download_button(
            "Download All History JSON",
            data=export_history_json(),
            file_name="textguard_all_history.json",
            mime="application/json"
        )

else:
    st.info(
        "No records available for export."
    )


# -------------------------------
# COMMIT 48: CONFIRM CLEAR HISTORY
# -------------------------------

st.subheader(
    "Clear Comparison History"
)

st.warning(
    "Clearing history will permanently delete "
    "all saved comparison records."
)

confirm_clear = st.checkbox(
    "I understand that this action cannot be undone.",
    key="confirm_clear_history"
)

if st.button(
    "🗑️ Clear All History",
    disabled=not confirm_clear,
    type="secondary"
):

    clear_history()

    st.session_state[
        "confirm_clear_history"
    ] = False

    st.rerun()


# -------------------------------
# HISTORY ANALYTICS DASHBOARD
# -------------------------------

st.divider()
st.subheader(
    "📊 History Analytics Dashboard"
)

analytics = history_analytics()

if analytics["total"] > 0:

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Comparisons",
        analytics["total"]
    )

    c2.metric(
        "Average TF-IDF",
        f"{analytics['average_similarity']:.2f}%"
    )

    c3.metric(
        "Highest Similarity",
        f"{analytics['highest_similarity']:.2f}%"
    )

    c4.metric(
        "Lowest Similarity",
        f"{analytics['lowest_similarity']:.2f}%"
    )

    st.subheader(
        "Average Similarity by Method"
    )

    method_data = pd.DataFrame({
        "Method": [
            "TF-IDF",
            "Jaccard",
            "Bigram"
        ],
        "Average (%)": [
            analytics["average_similarity"],
            analytics["average_jaccard"],
            analytics["average_bigram"]
        ]
    }).set_index("Method")

    st.bar_chart(method_data)

    st.subheader(
        "Similarity Trend Across Comparisons"
    )

    trend_df = pd.DataFrame(
        analytics["trend"]
    ).set_index("Comparison")

    st.line_chart(
        trend_df[
            [
                "TF-IDF",
                "Jaccard",
                "Bigram"
            ]
        ]
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(
            "Similarity Level Distribution"
        )

        level_df = pd.DataFrame(
            list(
                analytics[
                    "level_distribution"
                ].items()
            ),
            columns=[
                "Level",
                "Count"
            ]
        ).set_index("Level")

        st.bar_chart(level_df)

    with c2:

        st.subheader(
            "Risk Level Distribution"
        )

        risk_df = pd.DataFrame(
            list(
                analytics[
                    "risk_distribution"
                ].items()
            ),
            columns=[
                "Risk",
                "Count"
            ]
        ).set_index("Risk")

        st.bar_chart(risk_df)

    st.metric(
        "Total Similar Sentences Across Comparisons",
        analytics["total_similar_sentences"]
    )

    st.caption(
        "These charts summarize saved comparison records."
    )

else:
    st.info(
        "Run document comparisons to generate analytics."
    )


st.divider()

st.caption(
    "TextGuard | NLP Document Similarity Analyzer"
)

st.caption(
    "Comparison history is stored locally "
    "in textguard.db."
)
