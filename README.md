# Fake News Detection

A small web app that reads a news article and says whether its wording looks closer to real reporting or to fake news. It uses TF-IDF features and logistic regression, trained on the WELFake dataset, and runs as a Flask app.

## Results

Evaluated on a 20% held-out test set (12,736 articles).

| Metric | Value |
|---|---|
| Accuracy | 95.27% |
| Precision / recall (real) | 0.954 / 0.959 |
| Precision / recall (fake) | 0.951 / 0.945 |

Confusion matrix: 6,674 real and 5,459 fake articles classified correctly, 284 real flagged as fake, 319 fake passed as real. The image is in `models/confusion_matrix.png`.

## How it works

1. Load WELFake (72,134 articles) and drop empty rows and duplicates, leaving 63,678.
2. Join title and body, lowercase, strip non-letters, remove stopwords, lemmatize.
3. Turn the text into TF-IDF vectors (unigrams and bigrams, 100,000 features).
4. Train a logistic regression classifier on an 80/20 stratified split.
5. Serve predictions through a Flask page. If the model is less than 60% sure, it says "Unclear".

In this dataset label 1 is the fake class (checked by reading sample titles), so the app maps 1 to Fake and 0 to Real.

## Run it yourself

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. Download the dataset from Kaggle (`saurabhshahane/fake-news-classification`) and set `DATA_PATH` at the top of `train.py` to your copy of `WELFake_Dataset.csv`.
2. Train the model (takes a few minutes, writes the `models/` folder):

```bash
python train.py
```

3. Start the app and open http://127.0.0.1:5000:

```bash
python app.py
```

The trained `.joblib` files are not stored in the repo, so run `train.py` once before starting the app.

## Project structure

```
train.py            data cleaning, training, evaluation
app.py              Flask app
templates/index.html  web page
models/             metrics and confusion matrix (model files are generated locally)
requirements.txt
```

## Limitations

The model learns wording and style, not facts. WELFake's fake articles tend to be loud and clickbait-like, while the real ones read like news-agency copy, so a well-written fake can slip through. Treat the result as a hint, not proof.