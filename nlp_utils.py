
import re
import csv
import io
from collections import Counter

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


def classify_similarity(score):
    return "High" if score >= 70 else "Moderate" if score >= 40 else "Low"


def read_txt_file(file):
    return file.getvalue().decode("utf-8-sig", errors="replace")


def read_pdf_file(file):
    return "\n".join(p.extract_text() or "" for p in PdfReader(file).pages)


def split_sentences(text):
    return [
        s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text)
        if s.strip()
    ]


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


def find_duplicate_sentences(text1, text2):
    normalized = {
        preprocess_text(s) for s in split_sentences(text2)
        if preprocess_text(s)
    }
    duplicates = []
    for sentence in split_sentences(text1):
        cleaned = preprocess_text(sentence)
        if cleaned and cleaned in normalized and sentence not in duplicates:
            duplicates.append(sentence)
    return duplicates


def sentence_similarity_analysis(text1, text2):
    results = []
    sentences2 = split_sentences(text2)

    for sentence in split_sentences(text1):
        best_score, best_match = 0.0, ""
        for candidate in sentences2:
            score = calculate_similarity(sentence, candidate)
            if score > best_score:
                best_score, best_match = score, candidate

        results.append({
            "sentence": sentence,
            "best_match": best_match,
            "similarity": round(best_score, 2)
        })
    return results


def get_similar_sentences(text1, text2, threshold=60):
    return [
        r for r in sentence_similarity_analysis(text1, text2)
        if r["similarity"] >= threshold
    ]


def plagiarism_risk_summary(text1, text2, threshold=70):
    results = sentence_similarity_analysis(text1, text2)
    if not results or not split_sentences(text2):
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


def sentence_similarity_matrix(text1, text2):
    return [
        [round(calculate_similarity(a, b), 2) for b in split_sentences(text2)]
        for a in split_sentences(text1)
    ]


def sentence_results_csv(results):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Document 1 Sentence", "Closest Document 2 Sentence", "Similarity (%)"])
    for result in results:
        writer.writerow([result["sentence"], result["best_match"], result["similarity"]])
    return output.getvalue().encode("utf-8-sig")


def generate_text_report(text1, text2, score, results, threshold=60):
    stats1 = document_statistics(text1)
    stats2 = document_statistics(text2)
    common = get_common_keywords(text1, text2)
    unique = get_unique_keywords(text1, text2)
    duplicates = find_duplicate_sentences(text1, text2)
    summary = plagiarism_risk_summary(text1, text2)

    lines = [
        "TEXTGUARD - DOCUMENT SIMILARITY REPORT",
        "=" * 50,
        f"Overall Similarity: {score:.2f}%",
        f"Similarity Level: {classify_similarity(score)}",
        "",
        "DOCUMENT 1 STATISTICS", str(stats1),
        "",
        "DOCUMENT 2 STATISTICS", str(stats2),
        "",
        "COMMON KEYWORDS", ", ".join(common) or "None",
        "",
        "UNIQUE KEYWORDS - DOCUMENT 1",
        ", ".join(unique["document1"]) or "None",
        "",
        "UNIQUE KEYWORDS - DOCUMENT 2",
        ", ".join(unique["document2"]) or "None",
        "",
        "EXACT DUPLICATE SENTENCES"
    ]
    lines.extend(duplicates if duplicates else ["None"])
    lines.extend(["", "SENTENCE SIMILARITY RESULTS"])

    for result in results:
        lines.extend([
            f"Sentence: {result['sentence']}",
            f"Closest Match: {result['best_match']}",
            f"Similarity: {result['similarity']:.2f}%",
            "-" * 40
        ])

    lines.extend([
        f"Selected Display Threshold: {threshold}%",
        f"Risk Level: {summary['risk']}",
        f"Flagged Sentences (70% threshold): {summary['flagged_sentences']}",
        f"Flagged Percentage: {summary['flagged_percentage']:.2f}%",
        "",
        "DISCLAIMER: Similarity does not prove plagiarism."
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


def similarity_coverage(text1, text2, threshold=60):
    forward = sentence_similarity_analysis(text1, text2)
    reverse = sentence_similarity_analysis(text2, text1)

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
