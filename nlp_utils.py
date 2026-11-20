
import re
import csv
import io
import json
import html
from collections import Counter
from datetime import datetime, timezone

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# pyrefly: ignore[missing-import]
from nltk.stem import PorterStemmer

# pyrefly: ignore[missing-import]
from pypdf import PdfReader


def preprocess_text(text, remove_stopwords=False, apply_stemming=False):
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    if remove_stopwords:
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    if apply_stemming:
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]
    return " ".join(words)


def calculate_similarity(text1, text2, remove_stopwords=False, apply_stemming=False):
    a = preprocess_text(text1, remove_stopwords, apply_stemming)
    b = preprocess_text(text2, remove_stopwords, apply_stemming)
    if not a or not b:
        return 0.0
    try:
        vectors = TfidfVectorizer().fit_transform([a, b])
        return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100)
    except ValueError:
        return 0.0


def jaccard_similarity(text1, text2, remove_stopwords=False, apply_stemming=False):
    a = set(preprocess_text(text1, remove_stopwords, apply_stemming).split())
    b = set(preprocess_text(text2, remove_stopwords, apply_stemming).split())
    return round(100 * len(a & b) / len(a | b), 2) if a and b else 0.0


def get_bigrams(text, remove_stopwords=False, apply_stemming=False):
    words = preprocess_text(text, remove_stopwords, apply_stemming).split()
    return list(zip(words, words[1:]))


def bigram_similarity(text1, text2, remove_stopwords=False, apply_stemming=False):
    a = set(get_bigrams(text1, remove_stopwords, apply_stemming))
    b = set(get_bigrams(text2, remove_stopwords, apply_stemming))
    return round(100 * len(a & b) / len(a | b), 2) if a and b else 0.0


def common_bigrams(text1, text2, top_n=20):
    common = set(get_bigrams(text1)) & set(get_bigrams(text2))
    return [" ".join(pair) for pair in sorted(common)[:top_n]]


def longest_common_word_sequence(text1, text2):
    a = preprocess_text(text1).split()
    b = preprocess_text(text2).split()
    previous = [0] * (len(b) + 1)
    best_length = 0
    best_end = 0

    for i, word1 in enumerate(a, 1):
        current = [0] * (len(b) + 1)
        for j, word2 in enumerate(b, 1):
            if word1 == word2:
                current[j] = previous[j - 1] + 1
                if current[j] > best_length:
                    best_length = current[j]
                    best_end = i
        previous = current

    return {
        "sequence": " ".join(a[best_end - best_length:best_end]),
        "length": best_length
    }


def classify_similarity(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def read_txt_file(file):
    return file.getvalue().decode("utf-8-sig", errors="replace")


def read_pdf_file(file):
    return "\n".join(page.extract_text() or "" for page in PdfReader(file).pages)


def split_sentences(text, min_words=1):
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", text)
        if s.strip()
    ]
    return [s for s in sentences if len(preprocess_text(s).split()) >= min_words]


def document_statistics(text):
    return {
        "words": len(text.split()),
        "characters": len(text),
        "unique_words": len(set(preprocess_text(text).split())),
        "sentences": len(split_sentences(text))
    }


def get_common_keywords(text1, text2):
    a = set(preprocess_text(text1, True, True).split())
    b = set(preprocess_text(text2, True, True).split())
    return sorted(a & b)


def get_unique_keywords(text1, text2):
    a = set(preprocess_text(text1, True, True).split())
    b = set(preprocess_text(text2, True, True).split())
    return {"document1": sorted(a - b), "document2": sorted(b - a)}


def find_duplicate_sentences(text1, text2, min_words=1):
    normalized = {preprocess_text(s) for s in split_sentences(text2, min_words)}
    duplicates = []

    for sentence in split_sentences(text1, min_words):
        cleaned = preprocess_text(sentence)
        if cleaned and cleaned in normalized and sentence not in duplicates:
            duplicates.append(sentence)

    return duplicates


def sentence_similarity_analysis(text1, text2, min_words=1):
    results = []
    candidates = split_sentences(text2, min_words)

    for sentence in split_sentences(text1, min_words):
        best_score = 0.0
        best_match = ""

        for candidate in candidates:
            score = calculate_similarity(sentence, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate

        results.append({
            "sentence": sentence,
            "best_match": best_match,
            "similarity": round(best_score, 2)
        })

    return results


def sort_sentence_results(results, order="Highest First"):
    return sorted(
        results,
        key=lambda result: result["similarity"],
        reverse=(order == "Highest First")
    )


def search_sentence_results(results, query):
    query = query.strip().casefold()

    if not query:
        return list(results)

    return [
        result for result in results
        if query in result["sentence"].casefold()
        or query in result["best_match"].casefold()
    ]


def get_similar_sentences(text1, text2, threshold=60, min_words=1):
    return [
        r for r in sentence_similarity_analysis(text1, text2, min_words)
        if r["similarity"] >= threshold
    ]


def plagiarism_risk_summary(text1, text2, threshold=70, min_words=1):
    results = sentence_similarity_analysis(text1, text2, min_words)

    if not results or not split_sentences(text2, min_words):
        return {
            "total_sentences": len(results),
            "flagged_sentences": 0,
            "flagged_percentage": 0.0,
            "risk": "Not Available"
        }

    flagged = sum(r["similarity"] >= threshold for r in results)
    percentage = flagged / len(results) * 100
    risk = "High" if percentage >= 60 else "Moderate" if percentage >= 30 else "Low"

    return {
        "total_sentences": len(results),
        "flagged_sentences": flagged,
        "flagged_percentage": round(percentage, 2),
        "risk": risk
    }


def similarity_distribution(results):
    counts = Counter({"High": 0, "Moderate": 0, "Low": 0})

    for result in results:
        counts[classify_similarity(result["similarity"])] += 1

    return dict(counts)


def statistics_comparison(text1, text2):
    return {
        "Document 1": document_statistics(text1),
        "Document 2": document_statistics(text2)
    }


def sentence_similarity_matrix(text1, text2, min_words=1):
    a = split_sentences(text1, min_words)
    b = split_sentences(text2, min_words)

    return [
        [round(calculate_similarity(x, y), 2) for y in b]
        for x in a
    ]


def sentence_results_csv(results):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Document 1 Sentence",
        "Closest Document 2 Sentence",
        "Similarity (%)"
    ])

    for result in results:
        writer.writerow([
            result["sentence"],
            result["best_match"],
            result["similarity"]
        ])

    return output.getvalue().encode("utf-8-sig")


def generate_text_report(text1, text2, score, results, threshold=60, min_words=1):
    longest = longest_common_word_sequence(text1, text2)

    lines = [
        "TEXTGUARD DOCUMENT SIMILARITY REPORT",
        "=" * 45,
        f"TF-IDF Similarity: {score:.2f}%",
        f"Jaccard Similarity: {jaccard_similarity(text1, text2):.2f}%",
        f"Bigram Similarity: {bigram_similarity(text1, text2):.2f}%",
        f"Similarity Level: {classify_similarity(score)}",
        f"Longest Common Sequence: {longest['sequence'] or 'None'}",
        f"Longest Sequence Length: {longest['length']} words",
        f"Minimum Sentence Length: {min_words} words",
        "",
        "DOCUMENT STATISTICS",
        str(statistics_comparison(text1, text2)),
        "",
        "COMMON KEYWORDS",
        ", ".join(get_common_keywords(text1, text2)) or "None",
        "",
        "UNIQUE KEYWORDS",
        str(get_unique_keywords(text1, text2)),
        "",
        "COMMON BIGRAMS",
        ", ".join(common_bigrams(text1, text2)) or "None",
        "",
        "EXACT DUPLICATES"
    ]

    lines.extend(find_duplicate_sentences(text1, text2, min_words) or ["None"])
    lines.extend(["", "SENTENCE RESULTS"])

    for result in results:
        lines.extend([
            f"Sentence: {result['sentence']}",
            f"Closest Match: {result['best_match']}",
            f"Similarity: {result['similarity']:.2f}%"
        ])

    lines.extend([
        "",
        f"Display Threshold: {threshold}%",
        f"Risk Summary (70% threshold): {plagiarism_risk_summary(text1, text2, 70, min_words)}",
        "Similarity is not proof of plagiarism."
    ])

    return "\n".join(lines)


def word_frequency(text, top_n=10):
    words = [
        word for word in preprocess_text(text, True).split()
        if len(word) > 1
    ]
    return dict(Counter(words).most_common(top_n))


def top_keyword_frequencies(text, top_n=10):
    return word_frequency(text, top_n)


def similarity_coverage(text1, text2, threshold=60, min_words=1):
    forward = sentence_similarity_analysis(text1, text2, min_words)
    reverse = sentence_similarity_analysis(text2, text1, min_words)

    def coverage(results):
        if not results:
            return 0.0

        matched = sum(r["similarity"] >= threshold for r in results)
        return round(matched / len(results) * 100, 2)

    return {
        "document1": coverage(forward),
        "document2": coverage(reverse),
        "matched1": sum(r["similarity"] >= threshold for r in forward),
        "matched2": sum(r["similarity"] >= threshold for r in reverse),
        "total1": len(forward),
        "total2": len(reverse)
    }


def highlight_sentence(sentence, score, threshold=60):
    safe_sentence = html.escape(sentence)

    if score >= 70:
        background, border = "#ffd6d6", "#dc2626"
    elif score >= 40:
        background, border = "#fff0c2", "#d97706"
    else:
        background, border = "#dcfce7", "#16a34a"

    if score < threshold:
        background, border = "#f1f5f9", "#94a3b8"

    return (
        f'<div style="background:{background};border-left:5px solid {border};'
        f'padding:12px;margin:8px 0;border-radius:6px;color:#111827;">'
        f'<strong>{score:.2f}% similarity</strong>'
        f'<p style="margin:6px 0 0 0;">{safe_sentence}</p>'
        f'</div>'
    )


def generate_html_report(text1, text2, score, results, threshold=60,
                         min_words=1, remove_stopwords=False,
                         apply_stemming=False):
    jaccard = jaccard_similarity(
        text1, text2, remove_stopwords, apply_stemming
    )
    bigram = bigram_similarity(
        text1, text2, remove_stopwords, apply_stemming
    )
    longest = longest_common_word_sequence(text1, text2)
    risk = plagiarism_risk_summary(text1, text2, 70, min_words)
    stats1 = document_statistics(text1)
    stats2 = document_statistics(text2)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rows = []

    for result in results:
        similarity = result["similarity"]
        color = (
            "#fee2e2" if similarity >= 70
            else "#fef3c7" if similarity >= 40
            else "#dcfce7"
        )
        if similarity < threshold:
            color = "#f1f5f9"

        rows.append(
            f'<tr style="background:{color}">'
            f'<td>{html.escape(result["sentence"])}</td>'
            f'<td>{html.escape(result["best_match"])}</td>'
            f'<td>{similarity:.2f}%</td>'
            f'</tr>'
        )

    table_rows = "\n".join(rows) if rows else (
        '<tr><td colspan="3">No sentences available.</td></tr>'
    )

    common_words = ", ".join(get_common_keywords(text1, text2)) or "None"
    common_phrases = ", ".join(common_bigrams(text1, text2)) or "None"

    def stats_table(stats):
        return "".join(
            f"<tr><th>{html.escape(key.replace('_', ' ').title())}</th>"
            f"<td>{value}</td></tr>"
            for key, value in stats.items()
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TextGuard Similarity Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 1100px;
    margin: 35px auto;
    padding: 20px;
    background: #f8fafc;
    color: #1e293b;
}}
header {{
    background: #172554;
    color: white;
    padding: 28px;
    border-radius: 12px;
}}
h1 {{ margin: 0 0 8px; }}
section {{
    background: white;
    padding: 22px;
    border-radius: 12px;
    margin-top: 22px;
    border: 1px solid #e2e8f0;
}}
.metrics {{
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
}}
.metric {{
    background: #eff6ff;
    padding: 18px;
    border-radius: 10px;
    flex: 1;
    min-width: 160px;
}}
.metric strong {{
    display: block;
    font-size: 25px;
    margin-top: 8px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 12px;
}}
th, td {{
    border: 1px solid #cbd5e1;
    padding: 12px;
    text-align: left;
    vertical-align: top;
    overflow-wrap: anywhere;
}}
th {{ background: #e2e8f0; }}
.note {{ color: #64748b; font-size: 14px; }}
@media print {{
    body {{ background: white; margin: 0; }}
    section {{ break-inside: avoid; }}
}}
</style>
</head>
<body>
<header>
<h1>TextGuard Similarity Report</h1>
<p>NLP Document Similarity &amp; Plagiarism Screening</p>
<p>Generated: {timestamp}</p>
</header>

<section>
<h2>Overall Similarity</h2>
<div class="metrics">
<div class="metric">TF-IDF Cosine<strong>{score:.2f}%</strong></div>
<div class="metric">Jaccard<strong>{jaccard:.2f}%</strong></div>
<div class="metric">Bigram<strong>{bigram:.2f}%</strong></div>
<div class="metric">Similarity Level<strong>{classify_similarity(score)}</strong></div>
</div>
</section>

<section>
<h2>Document Statistics</h2>
<h3>Document 1</h3>
<table>{stats_table(stats1)}</table>
<h3>Document 2</h3>
<table>{stats_table(stats2)}</table>
</section>

<section>
<h2>Matching Content</h2>
<p><strong>Longest Common Word Sequence:</strong>
{html.escape(longest["sequence"]) or "None"}</p>
<p><strong>Consecutive Matching Words:</strong> {longest["length"]}</p>
<p><strong>Common Keywords:</strong> {html.escape(common_words)}</p>
<p><strong>Common Bigrams:</strong> {html.escape(common_phrases)}</p>
</section>

<section>
<h2>Potential Plagiarism Risk</h2>
<p><strong>Risk:</strong> {risk["risk"]}</p>
<p><strong>Flagged Sentences:</strong>
{risk["flagged_sentences"]} / {risk["total_sentences"]}</p>
<p><strong>Flagged Percentage:</strong>
{risk["flagged_percentage"]:.2f}%</p>
<p class="note">Risk uses a fixed 70% sentence similarity threshold.</p>
</section>

<section>
<h2>Sentence Comparison</h2>
<p class="note">Minimum sentence length: {min_words} words.
Display threshold: {threshold}%.</p>
<table>
<thead>
<tr>
<th>Document 1 Sentence</th>
<th>Closest Document 2 Sentence</th>
<th>Similarity</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
<p class="note">Red: high similarity; yellow: moderate;
green: low; gray: below selected threshold.</p>
</section>

<section>
<h2>Disclaimer</h2>
<p>Text similarity is not proof of plagiarism.
Results require human review and source verification.</p>
</section>
</body>
</html>"""


def generate_json_report(text1, text2, score, results, threshold=60,
                         remove_stopwords=False, apply_stemming=False,
                         min_words=1):
    report = {
        "project": "TextGuard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_similarity": round(score, 2),
        "jaccard_similarity": jaccard_similarity(
            text1, text2, remove_stopwords, apply_stemming
        ),
        "bigram_similarity": bigram_similarity(
            text1, text2, remove_stopwords, apply_stemming
        ),
        "common_bigrams": common_bigrams(text1, text2),
        "longest_common_sequence": longest_common_word_sequence(text1, text2),
        "similarity_level": classify_similarity(score),
        "settings": {
            "remove_stopwords": remove_stopwords,
            "apply_stemming": apply_stemming,
            "sentence_threshold": threshold,
            "minimum_sentence_words": min_words,
            "risk_threshold": 70
        },
        "document_statistics": statistics_comparison(text1, text2),
        "common_keywords": get_common_keywords(text1, text2),
        "unique_keywords": get_unique_keywords(text1, text2),
        "exact_duplicates": find_duplicate_sentences(text1, text2, min_words),
        "sentence_similarity": results,
        "similarity_distribution": similarity_distribution(results),
        "similarity_matrix": sentence_similarity_matrix(text1, text2, min_words),
        "word_frequency": {
            "document1": word_frequency(text1),
            "document2": word_frequency(text2)
        },
        "coverage": similarity_coverage(text1, text2, threshold, min_words),
        "risk_summary": plagiarism_risk_summary(text1, text2, 70, min_words),
        "disclaimer": "Similarity does not prove plagiarism."
    }

    return json.dumps(report, indent=4, ensure_ascii=False)


def analysis_summary(text1, text2, score, threshold=60, min_words=1):
    results = sentence_similarity_analysis(text1, text2, min_words)
    coverage = similarity_coverage(text1, text2, threshold, min_words)

    return {
        "overall_similarity": round(score, 2),
        "jaccard_similarity": jaccard_similarity(text1, text2),
        "bigram_similarity": bigram_similarity(text1, text2),
        "longest_common_sequence": longest_common_word_sequence(text1, text2),
        "similarity_level": classify_similarity(score),
        "common_keywords": len(get_common_keywords(text1, text2)),
        "exact_duplicates": len(find_duplicate_sentences(text1, text2, min_words)),
        "similar_sentences": sum(r["similarity"] >= threshold for r in results),
        "document1_coverage": coverage["document1"],
        "document2_coverage": coverage["document2"],
        "risk": plagiarism_risk_summary(text1, text2, 70, min_words)["risk"]
    }
