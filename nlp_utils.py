
import re
import csv
import io
import json
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
    return [
        " ".join(pair)
        for pair in sorted(set(get_bigrams(text1)) & set(get_bigrams(text2)))[:top_n]
    ]


def longest_common_word_sequence(text1, text2):
    a = preprocess_text(text1).split()
    b = preprocess_text(text2).split()

    if not a or not b:
        return {"sequence": "", "length": 0}

    previous = [0] * (len(b) + 1)
    best_length = 0
    best_end = 0

    for i, word1 in enumerate(a, start=1):
        current = [0] * (len(b) + 1)
        for j, word2 in enumerate(b, start=1):
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
    return "High" if score >= 70 else "Moderate" if score >= 40 else "Low"


def read_txt_file(file):
    return file.getvalue().decode("utf-8-sig", errors="replace")


def read_pdf_file(file):
    return "\n".join(p.extract_text() or "" for p in PdfReader(file).pages)


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def document_statistics(text):
    return {
        "words": len(text.split()),
        "characters": len(text),
        "unique_words": len(set(preprocess_text(text).split())),
        "sentences": len(split_sentences(text))
    }


def get_common_keywords(text1, text2):
    return sorted(
        set(preprocess_text(text1, True, True).split()) &
        set(preprocess_text(text2, True, True).split())
    )


def get_unique_keywords(text1, text2):
    a = set(preprocess_text(text1, True, True).split())
    b = set(preprocess_text(text2, True, True).split())
    return {"document1": sorted(a - b), "document2": sorted(b - a)}


def find_duplicate_sentences(text1, text2):
    normalized = {preprocess_text(s) for s in split_sentences(text2)}
    duplicates = []
    for sentence in split_sentences(text1):
        cleaned = preprocess_text(sentence)
        if cleaned and cleaned in normalized and sentence not in duplicates:
            duplicates.append(sentence)
    return duplicates


def sentence_similarity_analysis(text1, text2):
    results = []
    candidates = split_sentences(text2)
    for sentence in split_sentences(text1):
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


def get_similar_sentences(text1, text2, threshold=60):
    return [r for r in sentence_similarity_analysis(text1, text2)
            if r["similarity"] >= threshold]


def plagiarism_risk_summary(text1, text2, threshold=70):
    results = sentence_similarity_analysis(text1, text2)
    if not results or not split_sentences(text2):
        return {"total_sentences": len(results), "flagged_sentences": 0,
                "flagged_percentage": 0.0, "risk": "Not Available"}
    flagged = sum(r["similarity"] >= threshold for r in results)
    percentage = flagged / len(results) * 100
    risk = "High" if percentage >= 60 else "Moderate" if percentage >= 30 else "Low"
    return {"total_sentences": len(results), "flagged_sentences": flagged,
            "flagged_percentage": round(percentage, 2), "risk": risk}


def similarity_distribution(results):
    counts = Counter({"High": 0, "Moderate": 0, "Low": 0})
    for r in results:
        counts[classify_similarity(r["similarity"])] += 1
    return dict(counts)


def statistics_comparison(text1, text2):
    return {"Document 1": document_statistics(text1),
            "Document 2": document_statistics(text2)}


def sentence_similarity_matrix(text1, text2):
    return [[round(calculate_similarity(a, b), 2) for b in split_sentences(text2)]
            for a in split_sentences(text1)]


def sentence_results_csv(results):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Document 1 Sentence", "Closest Document 2 Sentence", "Similarity (%)"])
    for r in results:
        writer.writerow([r["sentence"], r["best_match"], r["similarity"]])
    return output.getvalue().encode("utf-8-sig")


def generate_text_report(text1, text2, score, results, threshold=60):
    longest = longest_common_word_sequence(text1, text2)
    lines = [
        "TEXTGUARD DOCUMENT SIMILARITY REPORT", "=" * 45,
        f"TF-IDF Similarity: {score:.2f}%",
        f"Jaccard Similarity: {jaccard_similarity(text1, text2):.2f}%",
        f"Bigram Similarity: {bigram_similarity(text1, text2):.2f}%",
        f"Similarity Level: {classify_similarity(score)}",
        f"Longest Common Sequence: {longest['sequence'] or 'None'}",
        f"Longest Sequence Length: {longest['length']} words",
        "", "DOCUMENT STATISTICS", str(statistics_comparison(text1, text2)),
        "", "COMMON KEYWORDS", ", ".join(get_common_keywords(text1, text2)) or "None",
        "", "UNIQUE KEYWORDS", str(get_unique_keywords(text1, text2)),
        "", "COMMON BIGRAMS", ", ".join(common_bigrams(text1, text2)) or "None",
        "", "EXACT DUPLICATES"
    ]
    lines.extend(find_duplicate_sentences(text1, text2) or ["None"])
    lines.extend(["", "SENTENCE RESULTS"])
    for r in results:
        lines.extend([
            f"Sentence: {r['sentence']}",
            f"Closest Match: {r['best_match']}",
            f"Similarity: {r['similarity']:.2f}%"
        ])
    lines.extend([
        "", f"Display Threshold: {threshold}%",
        f"Risk Summary (70% threshold): {plagiarism_risk_summary(text1, text2)}",
        "Similarity is not proof of plagiarism."
    ])
    return "\n".join(lines)


def word_frequency(text, top_n=10):
    words = [w for w in preprocess_text(text, True).split() if len(w) > 1]
    return dict(Counter(words).most_common(top_n))


def top_keyword_frequencies(text, top_n=10):
    return word_frequency(text, top_n)


def similarity_coverage(text1, text2, threshold=60):
    forward = sentence_similarity_analysis(text1, text2)
    reverse = sentence_similarity_analysis(text2, text1)

    def coverage(results):
        if not results:
            return 0.0
        return round(sum(r["similarity"] >= threshold for r in results)
                     / len(results) * 100, 2)

    return {
        "document1": coverage(forward),
        "document2": coverage(reverse),
        "matched1": sum(r["similarity"] >= threshold for r in forward),
        "matched2": sum(r["similarity"] >= threshold for r in reverse),
        "total1": len(forward),
        "total2": len(reverse)
    }


def generate_json_report(text1, text2, score, results, threshold=60,
                         remove_stopwords=False, apply_stemming=False):
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
            "risk_threshold": 70
        },
        "document_statistics": statistics_comparison(text1, text2),
        "common_keywords": get_common_keywords(text1, text2),
        "unique_keywords": get_unique_keywords(text1, text2),
        "exact_duplicates": find_duplicate_sentences(text1, text2),
        "sentence_similarity": results,
        "similarity_distribution": similarity_distribution(results),
        "similarity_matrix": sentence_similarity_matrix(text1, text2),
        "word_frequency": {
            "document1": word_frequency(text1),
            "document2": word_frequency(text2)
        },
        "coverage": similarity_coverage(text1, text2, threshold),
        "risk_summary": plagiarism_risk_summary(text1, text2),
        "disclaimer": "Similarity does not prove plagiarism."
    }
    return json.dumps(report, indent=4, ensure_ascii=False)


def analysis_summary(text1, text2, score, threshold=60):
    results = sentence_similarity_analysis(text1, text2)
    coverage = similarity_coverage(text1, text2, threshold)
    return {
        "overall_similarity": round(score, 2),
        "jaccard_similarity": jaccard_similarity(text1, text2),
        "bigram_similarity": bigram_similarity(text1, text2),
        "longest_common_sequence": longest_common_word_sequence(text1, text2),
        "similarity_level": classify_similarity(score),
        "common_keywords": len(get_common_keywords(text1, text2)),
        "exact_duplicates": len(find_duplicate_sentences(text1, text2)),
        "similar_sentences": sum(r["similarity"] >= threshold for r in results),
        "document1_coverage": coverage["document1"],
        "document2_coverage": coverage["document2"],
        "risk": plagiarism_risk_summary(text1, text2)["risk"]
    }
