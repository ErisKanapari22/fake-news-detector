import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.data_preprocessing import clean_text
from src.feature_extracion import extract_features


def evaluate():
    project_root = Path(__file__).resolve().parent.parent
    file_news_path = next(project_root.rglob("news.csv"), None)
    file_models_path = next(project_root.rglob("fake_news_model.pkl"), None)

    if file_news_path is None:
        raise FileNotFoundError("news.csv file not found in this project")

    if file_models_path is None:
        raise FileNotFoundError("fake_news_model.pkl file not found in this project")

    df = pd.read_csv(file_news_path)
    df["clean_text"] = df["text"].fillna("").apply(clean_text)
    x, _ = extract_features(df["clean_text"])
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = joblib.load(file_models_path)

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
    report = classification_report(y_test, y_pred, labels=["FAKE", "REAL"])

    return acc, cm, report


if __name__ == "__main__":
    acc, cm, report = evaluate()

    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix (row=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(report)
