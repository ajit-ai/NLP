"""Tests for Example 12 - Amazon reviews sentiment (real dataset).

Skips gracefully when pandas is unavailable; uses the dataset if present,
otherwise the script's synthetic fallback keeps everything runnable.
"""

import pytest

pandas = pytest.importorskip("pandas")
sklearn = pytest.importorskip("sklearn")

from helpers import load_example  # noqa: E402

example_12 = load_example("12_amazon_reviews_sentiment")


@pytest.fixture(scope="module")
def df():
    return example_12.load_dataset(sample_size=800, random_state=42)


def test_expected_columns(df):
    assert {"text", "label"}.issubset(df.columns)


def test_no_short_texts(df):
    assert (df["text"].str.len() > 3).all()


def test_labels_are_integers(df):
    assert df["label"].dtype.kind in "iu"


def test_sample_size_respected(df):
    assert len(df) <= 800


@pytest.fixture(scope="module")
def accuracy():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    data = example_12.load_dataset(sample_size=800)
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"],
        test_size=0.2, random_state=42, stratify=data["label"],
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=20.0, class_weight="balanced")),
    ]).fit(X_train, y_train)
    return pipeline.score(X_test, np.asarray(y_test))


def test_accuracy_above_chance(accuracy):
    # 5 balanced classes -> chance is 20%; a real signal clears 25%
    assert accuracy > 0.25
