# pyrefly: ignore[missing-import]

import re
# pyrefly: ignore[missing-import]
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_similarity(text1, text2):
    cleaned1 = preprocess_text(text1)
    cleaned2 = preprocess_text(text2)

    if not cleaned1 or not cleaned2:
        return 0.0

    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([cleaned1, cleaned2])
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(score * 100)
    except ValueError:
        return 0.0


def classify_similarity(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Moderate"
    return "Low"


def read_txt_file(uploaded_file):
    return uploaded_file.getvalue().decode("utf-8-sig", errors="replace")


def read_pdf_file(uploaded_file):
    reader = PdfReader(uploaded_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def document_statistics(text):
    words = text.split()
    cleaned_words = preprocess_text(text).split()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]

    return {
        "words": len(words),
        "characters": len(text),
        "unique_words": len(set(cleaned_words)),
        "sentences": len(sentences)
    }