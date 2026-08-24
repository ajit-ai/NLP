"""Tests for Example 08 - from-scratch Naive Bayes classifier."""

import math

import pytest

from nlptoolkit.naive_bayes import NaiveBayesClassifier

SPAM = [
    "win a free iphone now",
    "cheap meds buy now",
    "you won a lottery prize claim now",
]
HAM = [
    "meeting tomorrow at noon",
    "please review the report",
    "lunch with the team tomorrow",
]
DOCS = SPAM + HAM
LABELS = ["spam"] * len(SPAM) + ["ham"] * len(HAM)


@pytest.fixture()
def clf():
    return NaiveBayesClassifier(alpha=1.0).fit(DOCS, LABELS)


class TestFit:
    def test_classes_sorted(self, clf):
        assert clf.classes_ == ["ham", "spam"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            NaiveBayesClassifier().fit(["a"], ["x", "y"])

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            NaiveBayesClassifier().fit([], [])

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            NaiveBayesClassifier(alpha=0)

    def test_vocabulary_built(self, clf):
        assert "free" in clf.vocabulary_
        assert "meeting" in clf.vocabulary_


class TestPredict:
    @pytest.mark.parametrize("text", [
        "free lottery prize win now",
        "buy cheap iphone now",
    ])
    def test_spammy_text(self, clf, text):
        assert clf.predict(text) == "spam"

    def test_hammy_text(self, clf):
        assert clf.predict("please review the meeting notes") == "ham"

    def test_training_set_perfect(self, clf):
        assert all(clf.predict(d) == y for d, y in zip(DOCS, LABELS))

    def test_log_proba_returns_both_classes(self, clf):
        scores = clf.predict_log_proba("free prize")
        assert set(scores) == {"spam", "ham"}
        assert scores["spam"] > scores["ham"]

    def test_proba_is_normalized(self, clf):
        proba = clf.predict_proba("free lottery winner")
        assert sum(proba.values()) == pytest.approx(1.0)
        assert proba["spam"] > 0.5

    def test_unseen_words_still_predict(self, clf):
        # only prior knowledge remains; must not crash
        assert clf.predict("zzz qqq unknown") in clf.classes_

    def test_laplace_smoothing_avoids_zero(self, clf):
        scores = clf.predict_log_proba("zzz")
        assert all(math.isfinite(v) for v in scores.values())
