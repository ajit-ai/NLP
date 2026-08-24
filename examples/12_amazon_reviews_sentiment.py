"""Example 12 - Amazon review sentiment analysis on the real dataset.

Uses ``context/Amazon-Product-Reviews-Sentiment-Analysis-in-Python-Dataset.csv``
(the same dataset as the notebook).  Trains a TF-IDF + Logistic Regression
pipeline and reports held-out accuracy.  The script degrades gracefully
to a synthetic sample when the CSV is not present, so it can run anywhere.

Run:  python examples/12_amazon_reviews_sentiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "context" / "Amazon-Product-Reviews-Sentiment-Analysis-in-Python-Dataset.csv"


def load_dataset(sample_size: int = 4000, random_state: int = 42) -> pd.DataFrame:
    """Load the Amazon reviews dataset (or a synthetic fallback)."""
    if DATASET.exists():
        df = pd.read_csv(DATASET)
    else:
        print(f"[warn] {DATASET.name} not found - using a small synthetic sample")
        df = _synthetic_reviews()

    # standardize to columns: text (str), label (0=negative .. 4=positive)
    df = df.rename(columns={df.columns[0]: "text", df.columns[1]: "label"})
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() > 3]
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df


def _synthetic_reviews() -> pd.DataFrame:
    good = ["great product love it", "excellent quality works well",
            "amazing value very happy", "best purchase ever made"]
    bad = ["terrible quality broke fast", "awful waste of money",
           "poor design never again", "worst product do not buy"]
    rows = []
    for i in range(200):
        for t in good:
            rows.append((t + f" item {i}", 5))
        for t in bad:
            rows.append((t + f" item {i}", 1))
    return pd.DataFrame(rows, columns=["text", "label"])


def main() -> None:
    df = load_dataset()
    print(f"Dataset: {len(df)} reviews, class distribution:")
    print(df["label"].value_counts().sort_index().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=20.0, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    accuracy = pipeline.score(X_test, y_test)
    print(f"\nHeld-out accuracy: {accuracy:.3f}\n")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    demo = ["this product is absolutely wonderful", "broke immediately, total waste"]
    print("Demo predictions:", list(zip(demo, pipeline.predict(demo))))


if __name__ == "__main__":
    main()
