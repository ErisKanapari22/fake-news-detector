# Fake News Detector (NLP + Machine Learning)

A simple **Fake News Detector** that classifies news articles as **FAKE** or **REAL** using classic NLP + ML:
- Text cleaning (NLP preprocessing)
- Feature extraction with **TF-IDF**
- Model training with **Logistic Regression**
- Evaluation with **Accuracy, Confusion Matrix, Classification Report**
- Optional model comparison with **Naive Bayes**

## Project Structure
fake-news-detector/ <br>
├── data/ <br>
│ └── raw/ <br>
│ └── news.csv <br>
├── models/ <br>
│ ├── fake_news_model.pkl <br>
│ └── vectorizer.pkl <br>
├── notebooks/ <br>
│ └── notebook.ipynb <br>
├── src/ <br>
│ ├── data_preprocessing.py <br>
│ ├── feature_extraction.py <br>
│ ├── train_model.py <br>
│ ├── evaluate_model.py <br>
│ └── compare_models.py <br>
├── main.py <br>
├── requirements.txt <br>
└── README.md <br>

## Dataset

The dataset contains news articles labeled as:
- `FAKE`
- `REAL`

Example columns:
- `text` (news content)
- `label` (FAKE/REAL)

> Place your dataset file at: `data/raw/news.csv`

## Installation

### 1) Create & activate virtual environment (recommended)

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```
**Mac/Linux**
```
python3 -m venv .venv
source .venv/bin/activate
```

**Install dependencies**

```commandline
pip install -r requirements.txt
```
