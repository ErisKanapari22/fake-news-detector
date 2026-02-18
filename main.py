import joblib


def load_model():
    model = joblib.load("models/fake_news_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer


def predict_news(text):
    model, vectorizer = load_model()
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

if __name__ == "__main__":
    sample_news = """
    Scientists announce a new breakthrough in renewable energy technology
    that could significantly reduce carbon emissions worldwide.
    """
    result = predict_news(sample_news)
    print(f"Predict: {result}")
