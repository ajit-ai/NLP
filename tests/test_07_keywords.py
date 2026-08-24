"""Tests for Example 09 - keyword extraction."""

import pytest

from nlptoolkit.keywords import frequency_keywords, keyword_summary, rake_keywords

TEXT = (
    "Natural language processing enables computers to understand human "
    "language. Deep learning models power modern language systems."
)


class TestFrequencyKeywords:
    def test_returns_top_k(self):
        assert len(frequency_keywords(TEXT, top_k=5)) == 5

    def test_stopwords_excluded(self):
        words, _ = zip(*frequency_keywords(TEXT, top_k=10))
        assert "the" not in words
        assert "to" not in words

    def test_top_score_is_normalized_to_one(self):
        result = frequency_keywords(TEXT, top_k=3)
        assert result[0][1] == pytest.approx(1.0)

    def test_repeated_word_ranks_first(self):
        assert frequency_keywords("language language models", top_k=1)[0][0] == "language"

    def test_empty_text(self):
        assert frequency_keywords("", top_k=5) == []

    def test_all_stopword_text(self):
        assert frequency_keywords("the of and", top_k=5) == []

    def test_scores_sorted_descending(self):
        scores = [s for _, s in frequency_keywords(TEXT, top_k=8)]
        assert scores == sorted(scores, reverse=True)


class TestRakeKeywords:
    def test_phrases_exclude_stopwords(self):
        phrases, _ = zip(*rake_keywords(TEXT, top_k=10))
        stop = {"the", "to", "and", "of"}
        for phrase in phrases:
            assert not set(phrase.split()) & stop

    def test_scores_are_floats_sorted_descending(self):
        result = rake_keywords(TEXT, top_k=10)
        scores = [s for _, s in result]
        assert all(isinstance(s, float) for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_deterministic_ordering(self):
        assert rake_keywords(TEXT, top_k=8) == rake_keywords(TEXT, top_k=8)

    def test_empty_text(self):
        assert rake_keywords("", top_k=3) == []


def test_keyword_summary_returns_both_strategies():
    summary = keyword_summary(TEXT, top_k=4)
    assert set(summary) == {"frequency", "rake"}
    assert len(summary["frequency"]) <= 4
    assert len(summary["rake"]) <= 4
