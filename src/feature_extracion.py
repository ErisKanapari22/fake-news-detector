from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features(texts):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    x = vectorizer.fit_transform(texts)

    return x, vectorizer
