"""Tests for Example 11 - scikit-learn sentiment pipeline."""

import pytest

sklearn = pytest.importorskip("sklearn")

from helpers import load_example  # noqa: E402

example_11 = load_example("11_sklearn_sentiment")


@pytest.fixture(scope="module")
def model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    texts, labels = example_11.build_dataset()
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ]).fit(texts, labels)


def test_dataset_balanced():
    texts, labels = example_11.build_dataset()
    assert labels.count(1) == labels.count(0)
    assert len(texts) == len(labels) == 20


def test_dataset_all_strings():
    texts, _ = example_11.build_dataset()
    assert all(isinstance(t, str) and t for t in texts)


def test_training_accuracy_is_perfect(model):
    texts, labels = example_11.build_dataset()
    assert model.score(texts, labels) == 1.0


def test_predicts_positive_review(model):
    assert model.predict(["amazing wonderful excellent device"])[0] == 1


def test_predicts_negative_review(model):
    assert model.predict(["terrible awful broke immediately"])[0] == 0


def test_probabilities_valid(model):
    proba = model.predict_proba(["great product"])[0]
    assert len(proba) == 2
    assert sum(proba) == pytest.approx(1.0)


def test_cross_validation_beats_chance():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    texts, labels = example_11.build_dataset()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    scores = cross_val_score(pipeline, texts, labels, cv=3)
    assert (scores > 0.5).all()
