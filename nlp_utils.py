import re


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def calculate_similarity(text1, text2):
    words1 = set(preprocess_text(text1).split())
    words2 = set(preprocess_text(text2).split())

    union = words1.union(words2)
    intersection = words1.intersection(words2)

    if not union:
        return 0.0

    return len(intersection) / len(union) * 100