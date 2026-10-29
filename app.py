import re
from functools import lru_cache
from pathlib import Path

import joblib
import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ("stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

MODEL_DIR = Path(__file__).parent / "models"
vectorizer = joblib.load(MODEL_DIR / "tfidf.joblib")
model = joblib.load(MODEL_DIR / "model.joblib")

STOP = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# In WELFake, label 1 turned out to be the fake articles
LABELS = {0: "Real", 1: "Fake"}
MIN_WORDS = 5
UNSURE_BELOW = 0.60


@lru_cache(maxsize=None)
def lemma(word):
    return lemmatizer.lemmatize(word)


def clean(text):
    # Keep this identical to clean() in train.py, or predictions drift
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return " ".join(lemma(w) for w in text.split() if w not in STOP and len(w) > 2)


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    result = None

    if request.method == "POST":
        text = request.form.get("news", "").strip()
        cleaned = clean(text)

        if len(cleaned.split()) < MIN_WORDS:
            result = {"error": "That is too short to judge. Paste a few full sentences."}
        else:
            proba = model.predict_proba(vectorizer.transform([cleaned]))[0]
            best = int(proba.argmax())
            confidence = float(proba[best])
            label = LABELS[best] if confidence >= UNSURE_BELOW else "Unclear"
            result = {"label": label, "confidence": round(confidence * 100, 1)}

    return render_template("index.html", result=result, text=text)


if __name__ == "__main__":
    app.run(debug=False)