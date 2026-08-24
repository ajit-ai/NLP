"""Example 11 - Sentiment analysis with scikit-learn pipelines.

Compares Multinomial Naive Bayes and Logistic Regression on a small
hand-labelled product-review corpus using TF-IDF word features and
cross-validation.

Run:  python examples/11_sklearn_sentiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

POSITIVE = [
    "I love this product it works perfectly",
    "excellent quality absolutely wonderful",
    "great value for money very happy",
    "fantastic experience would buy again",
    "amazing sound quality love it",
    "works great exceeded my expectations",
    "superb build quality and fast shipping",
    "best purchase I have made this year",
    "wonderful design and easy to use",
    "highly recommend this to everyone",
]
NEGATIVE = [
    "terrible product stopped working in a week",
    "awful quality complete waste of money",
    "very disappointed it broke immediately",
    "horrible customer service never again",
    "poor battery life and cheap materials",
    "worst purchase I have ever made",
    "broke after two days total garbage",
    "regret buying this piece of junk",
    "does not work at all avoid it",
    "cheap plastic feel very unhappy",
]


def build_dataset():
    texts = POSITIVE + NEGATIVE
    labels = [1] * len(POSITIVE) + [0] * len(NEGATIVE)
    return texts, labels


def main() -> None:
    texts, labels = build_dataset()

    models = {
        "MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", MultinomialNB(alpha=0.5)),
        ]),
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, C=5.0)),
        ]),
    }

    print("3-fold cross-validation accuracy:")
    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, texts, labels, cv=3, scoring="accuracy")
        print(f"  {name}: {scores.mean():.3f} +/- {scores.std():.3f}")

    # final fit + demo predictions
    best = models["LogisticRegression"].fit(texts, labels)
    for review in [
        "this is an amazing fantastic device",
        "horrible awful waste of my money",
    ]:
        pred = best.predict([review])[0]
        proba = best.predict_proba([review])[0]
        sentiment = "positive" if pred == 1 else "negative"
        print(f"  {review!r} -> {sentiment} (p={proba[pred]:.2f})")


if __name__ == "__main__":
    main()
