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

    vectorizer = TfidfVectorizer()

    try:
        vectors = vectorizer.fit_transform(
            [cleaned1, cleaned2]
        )
    except ValueError:
        return 0.0

    score = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return float(score * 100)


def classify_similarity(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Moderate"
    else:
        return "Low"


def read_txt_file(uploaded_file):
    try:
        return uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="replace"
        )


def read_pdf_file(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)