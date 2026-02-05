import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(
        word for word in text.split() if word not in stop_words
    )
    return text

sample_text = "BREAKING NEWS!!! Trump wins 2020 election 😱😱"

print(clean_text(sample_text))


