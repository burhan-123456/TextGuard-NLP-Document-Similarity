
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# pyrefly: ignore[missing-import]
from pypdf import PdfReader


def preprocess_text(text, remove_stopwords=False):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
        text = " ".join(words)

    return text


def calculate_similarity(text1, text2):
    a = preprocess_text(text1)
    b = preprocess_text(text2)
    if not a or not b:
        return 0.0
    try:
        vectors = TfidfVectorizer().fit_transform([a, b])
        return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100)
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


def document_statistics(text):
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {
        "words": len(text.split()),
        "characters": len(text),
        "unique_words": len(set(preprocess_text(text).split())),
        "sentences": len(sentences)
    }
