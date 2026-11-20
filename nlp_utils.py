
import re
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
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(score * 100)
    except ValueError:
        return 0.0


def classify_similarity(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def read_txt_file(file):
    return file.getvalue().decode("utf-8-sig", errors="replace")


def read_pdf_file(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def split_sentences(text):
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", text)
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

    return {
        "document1": sorted(a - b),
        "document2": sorted(b - a)
    }


def find_duplicate_sentences(text1, text2):
    normalized = {
        preprocess_text(s)
        for s in split_sentences(text2)
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
        best_score = 0.0
        best_match = ""

        for candidate in sentences2:
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


def get_similar_sentences(text1, text2, threshold=60):
    return [
        result
        for result in sentence_similarity_analysis(text1, text2)
        if result["similarity"] >= threshold
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

    if percentage >= 60:
        risk = "High"
    elif percentage >= 30:
        risk = "Moderate"
    else:
        risk = "Low"

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
    sentences1 = split_sentences(text1)
    sentences2 = split_sentences(text2)

    matrix = []

    for sentence1 in sentences1:
        row = []

        for sentence2 in sentences2:
            score = calculate_similarity(sentence1, sentence2)
            row.append(round(score, 2))

        matrix.append(row)

    return matrix
