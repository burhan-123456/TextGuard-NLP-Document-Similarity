def calculate_similarity(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    union = words1.union(words2)
    intersection = words1.intersection(words2)

    if not union:
        return 0.0

    similarity = len(intersection) / len(union)

    return similarity * 100