import json
import re
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

DATA_PATH = Path(r"E:\Dev\Datasets\WELFake\WELFake_Dataset.csv")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

for pkg in ("stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

STOP = set(stopwords.words("english"))
LEMMA = WordNetLemmatizer()


@lru_cache(maxsize=None)
def lemma(word):
    return LEMMA.lemmatize(word)


def clean(text):
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return " ".join(lemma(w) for w in text.split() if w not in STOP and len(w) > 2)


def main():
    df = pd.read_csv(DATA_PATH)
    print("Raw shape:", df.shape)
    print("Missing values:\n", df.isna().sum())

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df = df[(df["title"] + df["text"]).str.strip() != ""]
    df = df.drop_duplicates(subset=["title", "text"])
    print("Rows after cleaning:", df.shape)
    print("Label counts:\n", df["label"].value_counts())

    for lab in sorted(df["label"].unique()):
        print(f"\nSample titles for label {lab}:")
        print(df[df["label"] == lab]["title"].head(3).to_string(index=False))

    print("\nCleaning text (this can take 3-8 minutes)...")
    df["content"] = (df["title"] + " " + df["text"]).map(clean)

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vec = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), min_df=3)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, y_train)

    pred = model.predict(Xte)
    acc = accuracy_score(y_test, pred)
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, pred, digits=4))

    report = classification_report(y_test, pred, output_dict=True)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    ConfusionMatrixDisplay.from_estimator(model, Xte, y_test)
    plt.title("Confusion Matrix")
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")

    joblib.dump(vec, OUT_DIR / "tfidf.joblib")
    joblib.dump(model, OUT_DIR / "model.joblib")
    print("Saved: models/tfidf.joblib, model.joblib, metrics.json, confusion_matrix.png")


if __name__ == "__main__":
    main()