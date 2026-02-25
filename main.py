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
    A new smartphone app claims it can read human thoughts
    and predict future events with 100 percent accuracy.
    Millions have already downloaded it overnight.
    """ # Real news

    sample_news1 = """ 
    The United Nations held a meeting on climate policy this week,
    where member states discussed new commitments to reduce carbon emissions.
    Several countries announced updated targets for renewable energy adoption
    over the next decade.
    """ # Real news

    result = predict_news(sample_news1)
    print(f"Predict: {result}")
