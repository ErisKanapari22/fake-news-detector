import pandas as pd
from pathlib import Path

from scipy.ndimage import label
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.data_preprocessing import clean_text
from src.feature_extracion import extract_features

def compare():
    project_root = Path(__file__).resolve().parent.parent
    file_news_path = next(project_root.rglob("news.csv"), None)

    if file_news_path is None:
        raise FileNotFoundError("news.csv is not in the project")

    df = pd.read_csv(file_news_path)
    df["text_clean"] = df["text"].fillna("").apply(clean_text)

    x, _ = extract_features(df["text_clean"])
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "MultinominalNB": MultinomialNB(),
    }

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
        report = classification_report(y_test, y_pred, labels=["FAKE", "REAL"])

        print("\n" + "=" * 60)
        print(f"Model: {name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Confusion Matrix: (row=true), (cols=pred): ")
        print(cm)
        print(f"Classification Report: ")
        print(report)

if __name__ == "__main__":
    compare()














