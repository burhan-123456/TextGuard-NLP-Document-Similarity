
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
    return round(len(a & b) / len(a | b) * 100, 2) if a and b else 0.0


def get_bigrams(text, remove_stopwords=False, apply_stemming=False):
    words = preprocess_text(text, remove_stopwords, apply_stemming).split()
    return list(zip(words, words[1:]))


def bigram_similarity(text1, text2, remove_stopwords=False, apply_stemming=False):
    a = set(get_bigrams(text1, remove_stopwords, apply_stemming))
    b = set(get_bigrams(text2, remove_stopwords, apply_stemming))
    return round(len(a & b) / len(a | b) * 100, 2) if a and b else 0.0


def common_bigrams(text1, text2, top_n=20):
    common = set(get_bigrams(text1)) & set(get_bigrams(text2))
    return [" ".join(pair) for pair in sorted(common)[:top_n]]


def longest_common_word_sequence(text1, text2):
    a = preprocess_text(text1).split()
    b = preprocess_text(text2).split()
    previous = [0] * (len(b) + 1)
    best_length, best_end = 0, 0
    for i, word1 in enumerate(a, 1):
        current = [0] * (len(b) + 1)
        for j, word2 in enumerate(b, 1):
            if word1 == word2:
                current[j] = previous[j - 1] + 1
                if current[j] > best_length:
                    best_length, best_end = current[j], i
        previous = current
    return {
        "sequence": " ".join(a[best_end - best_length:best_end]),
        "length": best_length
    }


def classify_similarity(score):
    return "High" if score >= 70 else "Moderate" if score >= 40 else "Low"


def read_txt_file(file):
    return file.getvalue().decode("utf-8-sig", errors="replace")


def read_pdf_file(file):
    return "\n".join(page.extract_text() or "" for page in PdfReader(file).pages)


def split_sentences(text, min_words=1):
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text)
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
    return list(dict.fromkeys(
        s for s in split_sentences(text1, min_words)
        if preprocess_text(s) and preprocess_text(s) in normalized
    ))


def sentence_similarity_analysis(text1, text2, min_words=1):
    results = []
    candidates = split_sentences(text2, min_words)
    for sentence in split_sentences(text1, min_words):
        best_score, best_match = 0.0, ""
        for candidate in candidates:
            score = calculate_similarity(sentence, candidate)
            if score > best_score:
                best_score, best_match = score, candidate
        results.append({
            "sentence": sentence,
            "best_match": best_match,
            "similarity": round(best_score, 2)
        })
    return results


def sort_sentence_results(results, order="Highest First"):
    return sorted(
        results, key=lambda r: r["similarity"],
        reverse=(order == "Highest First")
    )


def search_sentence_results(results, query):
    query = query.strip().casefold()
    return [
        r for r in results
        if query in r["sentence"].casefold()
        or query in r["best_match"].casefold()
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
    for r in results:
        counts[classify_similarity(r["similarity"])] += 1
    return dict(counts)


def statistics_comparison(text1, text2):
    return {
        "Document 1": document_statistics(text1),
        "Document 2": document_statistics(text2)
    }


def sentence_similarity_matrix(text1, text2, min_words=1):
    return [
        [round(calculate_similarity(a, b), 2)
         for b in split_sentences(text2, min_words)]
        for a in split_sentences(text1, min_words)
    ]


def sentence_results_csv(results):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Document 1 Sentence", "Closest Match", "Similarity (%)"])
    for r in results:
        writer.writerow([r["sentence"], r["best_match"], r["similarity"]])
    return output.getvalue().encode("utf-8-sig")


def generate_text_report(text1, text2, score, results, threshold=60, min_words=1):
    lines = [
        "TEXTGUARD SIMILARITY REPORT",
        f"TF-IDF Similarity: {score:.2f}%",
        f"Jaccard Similarity: {jaccard_similarity(text1, text2):.2f}%",
        f"Bigram Similarity: {bigram_similarity(text1, text2):.2f}%",
        f"Similarity Level: {classify_similarity(score)}",
        f"Statistics: {statistics_comparison(text1, text2)}",
        f"Common Keywords: {get_common_keywords(text1, text2)}",
        f"Unique Keywords: {get_unique_keywords(text1, text2)}",
        f"Common Bigrams: {common_bigrams(text1, text2)}",
        f"Longest Sequence: {longest_common_word_sequence(text1, text2)}",
        f"Exact Duplicates: {find_duplicate_sentences(text1, text2, min_words)}",
        f"Risk Summary: {plagiarism_risk_summary(text1, text2, 70, min_words)}",
        f"Display Threshold: {threshold}%",
        "",
        "SENTENCE COMPARISON"
    ]
    for r in results:
        lines.extend([
            f"Sentence: {r['sentence']}",
            f"Closest Match: {r['best_match']}",
            f"Similarity: {r['similarity']:.2f}%",
            ""
        ])
    lines.append("Similarity does not prove plagiarism.")
    return "\n".join(lines)


def word_frequency(text, top_n=10):
    words = [w for w in preprocess_text(text, True).split() if len(w) > 1]
    return dict(Counter(words).most_common(top_n))


def top_keyword_frequencies(text, top_n=10):
    return word_frequency(text, top_n)


def similarity_coverage(text1, text2, threshold=60, min_words=1):
    forward = sentence_similarity_analysis(text1, text2, min_words)
    reverse = sentence_similarity_analysis(text2, text1, min_words)

    def coverage(results):
        return round(
            sum(r["similarity"] >= threshold for r in results) /
            len(results) * 100, 2
        ) if results else 0.0

    return {
        "document1": coverage(forward),
        "document2": coverage(reverse),
        "matched1": sum(r["similarity"] >= threshold for r in forward),
        "matched2": sum(r["similarity"] >= threshold for r in reverse),
        "total1": len(forward),
        "total2": len(reverse)
    }


def highlight_sentence(sentence, score, threshold=60):
    if score < threshold:
        background, border = "#f1f5f9", "#94a3b8"
    elif score >= 70:
        background, border = "#ffd6d6", "#dc2626"
    elif score >= 40:
        background, border = "#fff0c2", "#d97706"
    else:
        background, border = "#dcfce7", "#16a34a"

    return (
        f'<div style="background:{background};border-left:5px solid {border};'
        f'padding:12px;margin:8px 0;border-radius:6px;color:#111827;">'
        f'<strong>{score:.2f}% similarity</strong>'
        f'<p>{html.escape(sentence)}</p></div>'
    )


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
        "longest_common_sequence": longest_common_word_sequence(text1, text2),
        "similarity_level": classify_similarity(score),
        "settings": {
            "remove_stopwords": remove_stopwords,
            "apply_stemming": apply_stemming,
            "sentence_threshold": threshold,
            "minimum_sentence_words": min_words
        },
        "document_statistics": statistics_comparison(text1, text2),
        "common_keywords": get_common_keywords(text1, text2),
        "unique_keywords": get_unique_keywords(text1, text2),
        "common_bigrams": common_bigrams(text1, text2),
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
    stats = statistics_comparison(text1, text2)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rows = []
    for r in results:
        color = (
            "#f1f5f9" if r["similarity"] < threshold else
            "#fee2e2" if r["similarity"] >= 70 else
            "#fef3c7" if r["similarity"] >= 40 else "#dcfce7"
        )
        rows.append(
            f'<tr style="background:{color}">'
            f'<td>{html.escape(r["sentence"])}</td>'
            f'<td>{html.escape(r["best_match"])}</td>'
            f'<td>{r["similarity"]:.2f}%</td></tr>'
        )

    rows_html = "\n".join(rows) or '<tr><td colspan="3">No matches</td></tr>'
    common = html.escape(", ".join(get_common_keywords(text1, text2)) or "None")
    phrase = html.escape(longest["sequence"] or "None")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TextGuard Report</title>
<style>
body {{font-family:Arial;max-width:1050px;margin:30px auto;padding:20px;
background:#f8fafc;color:#1e293b}}
header {{background:#172554;color:white;padding:25px;border-radius:12px}}
section {{background:white;padding:20px;margin-top:20px;border-radius:12px}}
table {{width:100%;border-collapse:collapse}}
th,td {{border:1px solid #cbd5e1;padding:12px;text-align:left;vertical-align:top;
overflow-wrap:anywhere}}
th {{background:#e2e8f0}}
</style>
</head>
<body>
<header><h1>TextGuard Similarity Report</h1><p>{timestamp}</p></header>
<section>
<h2>Overall Results</h2>
<p>TF-IDF: <strong>{score:.2f}%</strong></p>
<p>Jaccard: <strong>{jaccard:.2f}%</strong></p>
<p>Bigram: <strong>{bigram:.2f}%</strong></p>
<p>Level: <strong>{classify_similarity(score)}</strong></p>
</section>
<section>
<h2>Document Statistics</h2>
<p>Document 1: {html.escape(str(stats["Document 1"]))}</p>
<p>Document 2: {html.escape(str(stats["Document 2"]))}</p>
</section>
<section>
<h2>Matching Content</h2>
<p>Longest matching sequence: {phrase}</p>
<p>Matching words: {longest["length"]}</p>
<p>Common keywords: {common}</p>
</section>
<section>
<h2>Risk Summary</h2>
<p>Risk: {risk["risk"]}</p>
<p>Flagged: {risk["flagged_sentences"]}/{risk["total_sentences"]}</p>
<p>Flagged percentage: {risk["flagged_percentage"]:.2f}%</p>
</section>
<section>
<h2>Sentence Matches</h2>
<p>Display threshold: {threshold}% | Minimum words: {min_words}</p>
<table>
<tr><th>Document 1</th><th>Closest Document 2 Match</th><th>Similarity</th></tr>
{rows_html}
</table>
</section>
<section><p>Similarity does not prove plagiarism.</p></section>
</body>
</html>"""
