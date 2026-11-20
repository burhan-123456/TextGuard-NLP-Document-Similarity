
import re
import io
import csv
import json
import html
from collections import Counter
from datetime import datetime, timezone

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.metrics.pairwise import cosine_similarity

# pyrefly: ignore[missing-import]
from nltk.stem import PorterStemmer

# pyrefly: ignore[missing-import]
from pypdf import PdfReader


def preprocess_text(
    text,
    remove_stopwords=False,
    apply_stemming=False
):
    words = re.sub(
        r"[^\w\s]",
        " ",
        text.lower()
    ).split()

    if remove_stopwords:
        words = [
            w for w in words
            if w not in ENGLISH_STOP_WORDS
        ]

    if apply_stemming:
        stemmer = PorterStemmer()
        words = [
            stemmer.stem(w)
            for w in words
        ]

    return " ".join(words)


def calculate_similarity(
    a,
    b,
    remove_stopwords=False,
    apply_stemming=False
):
    a = preprocess_text(
        a, remove_stopwords, apply_stemming
    )
    b = preprocess_text(
        b, remove_stopwords, apply_stemming
    )

    if not a or not b:
        return 0.0

    try:
        vectors = TfidfVectorizer().fit_transform(
            [a, b]
        )
        return float(
            cosine_similarity(
                vectors[:1],
                vectors[1:]
            )[0, 0] * 100
        )
    except ValueError:
        return 0.0


def jaccard_similarity(
    a,
    b,
    remove_stopwords=False,
    apply_stemming=False
):
    x = set(
        preprocess_text(
            a, remove_stopwords, apply_stemming
        ).split()
    )
    y = set(
        preprocess_text(
            b, remove_stopwords, apply_stemming
        ).split()
    )

    if not x or not y:
        return 0.0

    return round(
        100 * len(x & y) / len(x | y),
        2
    )


def get_bigrams(
    text,
    remove_stopwords=False,
    apply_stemming=False
):
    words = preprocess_text(
        text, remove_stopwords, apply_stemming
    ).split()

    return list(zip(words, words[1:]))


def bigram_similarity(
    a,
    b,
    remove_stopwords=False,
    apply_stemming=False
):
    x = set(
        get_bigrams(
            a, remove_stopwords, apply_stemming
        )
    )
    y = set(
        get_bigrams(
            b, remove_stopwords, apply_stemming
        )
    )

    if not x or not y:
        return 0.0

    return round(
        100 * len(x & y) / len(x | y),
        2
    )


def common_bigrams(a, b, top_n=20):
    common = sorted(
        set(get_bigrams(a)) &
        set(get_bigrams(b))
    )

    return [
        " ".join(pair)
        for pair in common[:top_n]
    ]


def longest_common_word_sequence(a, b):
    a = preprocess_text(a).split()
    b = preprocess_text(b).split()

    previous = [0] * (len(b) + 1)
    best = 0
    end = 0

    for i, word in enumerate(a, 1):
        current = [0] * (len(b) + 1)

        for j, other in enumerate(b, 1):
            if word == other:
                current[j] = previous[j - 1] + 1

                if current[j] > best:
                    best = current[j]
                    end = i

        previous = current

    return {
        "sequence": " ".join(
            a[end - best:end]
        ),
        "length": best
    }


def classify_similarity(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def read_txt_file(file):
    return file.getvalue().decode(
        "utf-8-sig",
        errors="replace"
    )


def read_pdf_file(file):
    reader = PdfReader(file)

    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )


def split_sentences(text, min_words=1):
    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
        and len(
            preprocess_text(sentence).split()
        ) >= min_words
    ]


def document_statistics(text):
    return {
        "words": len(text.split()),
        "characters": len(text),
        "unique_words": len(
            set(preprocess_text(text).split())
        ),
        "sentences": len(split_sentences(text))
    }


def statistics_comparison(a, b):
    return {
        "Document 1": document_statistics(a),
        "Document 2": document_statistics(b)
    }


def get_common_keywords(a, b):
    x = set(
        preprocess_text(a, True, True).split()
    )
    y = set(
        preprocess_text(b, True, True).split()
    )

    return sorted(x & y)


def get_unique_keywords(a, b):
    x = set(
        preprocess_text(a, True, True).split()
    )
    y = set(
        preprocess_text(b, True, True).split()
    )

    return {
        "document1": sorted(x - y),
        "document2": sorted(y - x)
    }


def word_level_comparison(a, b):
    x = Counter(preprocess_text(a).split())
    y = Counter(preprocess_text(b).split())

    results = []

    for word in sorted(set(x) | set(y)):
        if word in x and word in y:
            status = "Common"
        elif word in x:
            status = "Document 1 Only"
        else:
            status = "Document 2 Only"

        results.append({
            "Word": word,
            "Document 1 Frequency": x[word],
            "Document 2 Frequency": y[word],
            "Status": status
        })

    return results


def find_duplicate_sentences(a, b, min_words=1):
    normalized = {
        preprocess_text(sentence)
        for sentence in split_sentences(
            b, min_words
        )
    }

    duplicates = []

    for sentence in split_sentences(
        a, min_words
    ):
        cleaned = preprocess_text(sentence)

        if (
            cleaned
            and cleaned in normalized
            and sentence not in duplicates
        ):
            duplicates.append(sentence)

    return duplicates


def sentence_similarity_analysis(
    a, b, min_words=1
):
    candidates = split_sentences(
        b, min_words
    )

    results = []

    for sentence in split_sentences(
        a, min_words
    ):
        best = 0.0
        match = ""

        for candidate in candidates:
            score = calculate_similarity(
                sentence, candidate
            )

            if score > best:
                best = score
                match = candidate

        results.append({
            "sentence": sentence,
            "best_match": match,
            "similarity": round(best, 2)
        })

    return results


def sort_sentence_results(
    results,
    order="Highest First"
):
    return sorted(
        results,
        key=lambda r: r["similarity"],
        reverse=order == "Highest First"
    )


def search_sentence_results(results, query):
    q = query.strip().casefold()

    return [
        r for r in results
        if q in r["sentence"].casefold()
        or q in r["best_match"].casefold()
    ]


def get_similar_sentences(
    a, b, threshold=60, min_words=1
):
    return [
        r
        for r in sentence_similarity_analysis(
            a, b, min_words
        )
        if r["similarity"] >= threshold
    ]


def plagiarism_risk_summary(
    a, b, threshold=70, min_words=1
):
    rows = sentence_similarity_analysis(
        a, b, min_words
    )

    if not rows or not split_sentences(
        b, min_words
    ):
        return {
            "total_sentences": len(rows),
            "flagged_sentences": 0,
            "flagged_percentage": 0.0,
            "risk": "Not Available"
        }

    flagged = sum(
        r["similarity"] >= threshold
        for r in rows
    )

    percent = round(
        100 * flagged / len(rows),
        2
    )

    if percent >= 60:
        risk = "High"
    elif percent >= 30:
        risk = "Moderate"
    else:
        risk = "Low"

    return {
        "total_sentences": len(rows),
        "flagged_sentences": flagged,
        "flagged_percentage": percent,
        "risk": risk
    }


def similarity_distribution(results):
    counts = Counter({
        "High": 0,
        "Moderate": 0,
        "Low": 0
    })

    for result in results:
        counts[
            classify_similarity(
                result["similarity"]
            )
        ] += 1

    return dict(counts)


def sentence_similarity_matrix(
    a, b, min_words=1
):
    return [
        [
            round(
                calculate_similarity(x, y),
                2
            )
            for y in split_sentences(
                b, min_words
            )
        ]
        for x in split_sentences(
            a, min_words
        )
    ]


def sentence_results_csv(results):
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Document 1 Sentence",
        "Closest Match",
        "Similarity (%)"
    ])

    for result in results:
        writer.writerow([
            result["sentence"],
            result["best_match"],
            result["similarity"]
        ])

    return output.getvalue().encode(
        "utf-8-sig"
    )


def word_frequency(text, top_n=10):
    words = [
        word
        for word in preprocess_text(
            text, True
        ).split()
        if len(word) > 1
    ]

    return dict(
        Counter(words).most_common(top_n)
    )


def top_keyword_frequencies(
    text, top_n=10
):
    return word_frequency(
        text, top_n
    )


def similarity_coverage(
    a, b, threshold=60, min_words=1
):
    forward = sentence_similarity_analysis(
        a, b, min_words
    )
    reverse = sentence_similarity_analysis(
        b, a, min_words
    )

    def percentage(rows):
        if not rows:
            return 0.0

        matched = sum(
            r["similarity"] >= threshold
            for r in rows
        )

        return round(
            100 * matched / len(rows),
            2
        )

    return {
        "document1": percentage(forward),
        "document2": percentage(reverse),
        "matched1": sum(
            r["similarity"] >= threshold
            for r in forward
        ),
        "matched2": sum(
            r["similarity"] >= threshold
            for r in reverse
        ),
        "total1": len(forward),
        "total2": len(reverse)
    }


def highlight_sentence(
    sentence, score, threshold=60
):
    if score < threshold:
        bg = "#f1f5f9"
        border = "#94a3b8"
    elif score >= 70:
        bg = "#ffd6d6"
        border = "#dc2626"
    elif score >= 40:
        bg = "#fff0c2"
        border = "#d97706"
    else:
        bg = "#dcfce7"
        border = "#16a34a"

    return (
        f'<div style="background:{bg};'
        f'border-left:5px solid {border};'
        f'padding:12px;margin:8px 0;'
        f'border-radius:6px;color:#111827">'
        f'<strong>{score:.2f}% similarity</strong>'
        f'<p>{html.escape(sentence)}</p>'
        f'</div>'
    )


def analysis_summary(
    a, b, score, threshold=60, min_words=1
):
    results = sentence_similarity_analysis(
        a, b, min_words
    )

    coverage = similarity_coverage(
        a, b, threshold, min_words
    )

    return {
        "overall_similarity": round(score, 2),
        "jaccard_similarity": jaccard_similarity(a, b),
        "bigram_similarity": bigram_similarity(a, b),
        "longest_common_sequence":
            longest_common_word_sequence(a, b),
        "similarity_level":
            classify_similarity(score),
        "common_keywords":
            len(get_common_keywords(a, b)),
        "exact_duplicates":
            len(find_duplicate_sentences(
                a, b, min_words
            )),
        "similar_sentences": sum(
            r["similarity"] >= threshold
            for r in results
        ),
        "document1_coverage":
            coverage["document1"],
        "document2_coverage":
            coverage["document2"],
        "risk": plagiarism_risk_summary(
            a, b, 70, min_words
        )["risk"]
    }


def create_history_entry(
    a,
    b,
    score,
    threshold=60,
    min_words=1,
    remove_stopwords=False,
    apply_stemming=False
):
    return {
        "Time (UTC)": datetime.now(
            timezone.utc
        ).isoformat(),
        "Document 1 Words": len(a.split()),
        "Document 2 Words": len(b.split()),
        "TF-IDF (%)": round(score, 2),
        "Jaccard (%)": jaccard_similarity(
            a, b, remove_stopwords, apply_stemming
        ),
        "Bigram (%)": bigram_similarity(
            a, b, remove_stopwords, apply_stemming
        ),
        "Level": classify_similarity(score),
        "Similar Sentences": len(
            get_similar_sentences(
                a, b, threshold, min_words
            )
        ),
        "Risk": plagiarism_risk_summary(
            a, b, 70, min_words
        )["risk"],
        "Threshold (%)": threshold,
        "Minimum Words": min_words,
        "Remove Stopwords": remove_stopwords,
        "Apply Stemming": apply_stemming
    }


def build_report_data(
    a,
    b,
    score,
    results,
    threshold=60,
    min_words=1,
    remove_stopwords=False,
    apply_stemming=False
):
    return {
        "project": "TextGuard",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "overall_similarity": round(score, 2),
        "jaccard_similarity": jaccard_similarity(
            a, b, remove_stopwords, apply_stemming
        ),
        "bigram_similarity": bigram_similarity(
            a, b, remove_stopwords, apply_stemming
        ),
        "similarity_level": classify_similarity(
            score
        ),
        "settings": {
            "remove_stopwords": remove_stopwords,
            "apply_stemming": apply_stemming,
            "sentence_threshold": threshold,
            "minimum_sentence_words": min_words
        },
        "document_statistics":
            statistics_comparison(a, b),
        "common_keywords":
            get_common_keywords(a, b),
        "unique_keywords":
            get_unique_keywords(a, b),
        "word_level_comparison":
            word_level_comparison(a, b),
        "common_bigrams":
            common_bigrams(a, b),
        "longest_common_sequence":
            longest_common_word_sequence(a, b),
        "exact_duplicates":
            find_duplicate_sentences(
                a, b, min_words
            ),
        "sentence_similarity": results,
        "similarity_distribution":
            similarity_distribution(results),
        "similarity_matrix":
            sentence_similarity_matrix(
                a, b, min_words
            ),
        "word_frequency": {
            "document1": word_frequency(a),
            "document2": word_frequency(b)
        },
        "coverage": similarity_coverage(
            a, b, threshold, min_words
        ),
        "risk_summary": plagiarism_risk_summary(
            a, b, 70, min_words
        ),
        "disclaimer":
            "Similarity does not prove plagiarism."
    }


def generate_json_report(
    a,
    b,
    score,
    results,
    threshold=60,
    remove_stopwords=False,
    apply_stemming=False,
    min_words=1
):
    data = build_report_data(
        a, b, score, results, threshold,
        min_words, remove_stopwords,
        apply_stemming
    )

    return json.dumps(
        data,
        indent=4,
        ensure_ascii=False
    )


def generate_text_report(
    a, b, score, results,
    threshold=60, min_words=1
):
    data = build_report_data(
        a, b, score, results,
        threshold, min_words
    )

    lines = [
        "TEXTGUARD SIMILARITY REPORT",
        "=" * 40
    ]

    lines.extend(
        f"{key}: {value}"
        for key, value in data.items()
        if key != "sentence_similarity"
    )

    lines.append("\nSENTENCE RESULTS")

    lines.extend(
        f"{r['similarity']:.2f}% | "
        f"{r['sentence']} | "
        f"{r['best_match']}"
        for r in results
    )

    return "\n".join(lines)


def generate_html_report(
    a,
    b,
    score,
    results,
    threshold=60,
    min_words=1,
    remove_stopwords=False,
    apply_stemming=False
):
    data = build_report_data(
        a, b, score, results,
        threshold, min_words,
        remove_stopwords, apply_stemming
    )

    details = "".join(
        f"<p><strong>"
        f"{html.escape(key.replace('_', ' ').title())}:"
        f"</strong> "
        f"{html.escape(str(value))}</p>"
        for key, value in data.items()
        if key not in (
            "sentence_similarity",
            "similarity_matrix",
            "word_level_comparison"
        )
    )

    rows = []

    for result in results:
        similarity = result["similarity"]

        if similarity < threshold:
            color = "#f1f5f9"
        elif similarity >= 70:
            color = "#fee2e2"
        elif similarity >= 40:
            color = "#fef3c7"
        else:
            color = "#dcfce7"

        rows.append(
            f'<tr style="background:{color}">'
            f'<td>{html.escape(result["sentence"])}</td>'
            f'<td>{html.escape(result["best_match"])}</td>'
            f'<td>{similarity:.2f}%</td>'
            f'</tr>'
        )

    table_rows = "".join(rows)

    if not table_rows:
        table_rows = (
            '<tr><td colspan="3">'
            'No matches'
            '</td></tr>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TextGuard Report</title>
<style>
body {{
    font-family: Arial;
    max-width: 1050px;
    margin: 30px auto;
    padding: 20px;
    background: #f8fafc;
    color: #1e293b;
}}
header {{
    background: #172554;
    color: white;
    padding: 25px;
    border-radius: 12px;
}}
section {{
    background: white;
    padding: 20px;
    margin-top: 20px;
    border-radius: 12px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
}}
th, td {{
    border: 1px solid #cbd5e1;
    padding: 12px;
    text-align: left;
    vertical-align: top;
    overflow-wrap: anywhere;
}}
th {{
    background: #e2e8f0;
}}
</style>
</head>
<body>
<header>
<h1>TextGuard Similarity Report</h1>
</header>

<section>
<h2>Analysis Summary</h2>
{details}
</section>

<section>
<h2>Sentence Comparison</h2>
<table>
<tr>
<th>Document 1</th>
<th>Closest Match</th>
<th>Similarity</th>
</tr>
{table_rows}
</table>
</section>

<section>
<p>Similarity does not prove plagiarism.</p>
</section>
</body>
</html>"""
