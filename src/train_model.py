import pandas as pd
import joblib
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data_preprocessing import clean_text
from src.feature_extracion import extract_features


def train():
    project_root = Path(__file__).resolve().parent.parent
    file_news_path = next(project_root.rglob("news.csv"))

    if file_news_path is None:
        raise FileNotFoundError("File news.csv not found")


    df = pd.read_csv(file_news_path)
    df['clean_text'] = df['text'].apply(clean_text)
    x, vectorizer = extract_features(df['clean_text'])
    y = df['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print("Model Trained Successfully!")
    print("Accuracy: ", acc)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODELS_DIR, "fake_news_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))

    print("Model saved in models/")


if __name__ == "__main__":
    train()
