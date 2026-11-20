import re
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