"""Tests for Example 04/05/06 - feature extraction (one-hot, BoW, TF-IDF,
n-grams)."""

import math

import pytest

from nlptoolkit.features import (
    CountVectorizer,
    OneHotEncoder,
    TfidfVectorizer,
    bigram_features,
    char_ngrams,
    ngrams,
)

DOCS = [
    "the cat sat on the mat",
    "the dog ate my homework",
]


class TestOneHot:
    def test_unit_vector(self):
        enc = OneHotEncoder().fit(DOCS)
        vec = enc.encode("cat")
        assert sum(vec) == 1.0
        assert vec[enc.vocab["cat"]] == 1.0

    def test_unknown_word_raises(self):
        enc = OneHotEncoder().fit(DOCS)
        with pytest.raises(KeyError):
            enc.encode("zebra")

    def test_vocab_size_matches_dimension(self):
        enc = OneHotEncoder().fit(["a b c"])
        assert len(enc.vocab) == 3
        assert len(enc.encode("a")) == 3


class TestCountVectorizer:
    def test_counts_are_correct(self):
        vec = CountVectorizer()
        matrix = vec.fit_transform(DOCS)
        idx_the = vec.vocabulary_["the"]
        assert matrix[0][idx_the] == 2
        assert matrix[1][idx_the] == 1

    def test_unknown_words_ignored(self):
        vec = CountVectorizer().fit(DOCS)
        row = vec.transform(["unicorn cat"])[0]
        assert row[vec.vocabulary_["cat"]] == 1
        assert sum(row) == 1

    def test_lowercase_normalization(self):
        vec = CountVectorizer().fit(["Hello hello WORLD"])
        # 'hello' and 'HELLO' collapse into one vocabulary entry
        assert len(vec.vocabulary_) == 2
        assert set(vec.vocabulary_) == {"hello", "world"}

    def test_transform_before_fit_empty_vocab(self):
        vec = CountVectorizer()
        assert vec.transform(["anything"]) == [[]]

    def test_row_sums_equal_token_counts(self):
        docs = ["one two two three three three"]
        vec = CountVectorizer()
        row = vec.fit_transform(docs)[0]
        assert sum(row) == 6


class TestTfidfVectorizer:
    def test_fit_builds_vocab_and_idf(self):
        tfidf = TfidfVectorizer().fit(DOCS)
        assert "cat" in tfidf.vocabulary_
        # a term in every doc should have lower idf than a rare term
        assert tfidf.idf_["the"] < tfidf.idf_["homework"]

    def test_rows_l2_normalized(self):
        matrix = TfidfVectorizer().fit_transform(DOCS)
        for row in matrix:
            norm = math.sqrt(sum(v * v for v in row))
            assert norm == pytest.approx(1.0, abs=1e-9)

    def test_common_word_gets_lower_weight(self):
        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(DOCS)
        idx_the = tfidf.vocabulary_["the"]
        idx_homework = tfidf.vocabulary_["homework"]
        row = matrix[1]
        assert row[idx_homework] > row[idx_the]

    def test_transform_requires_fit(self):
        with pytest.raises(RuntimeError):
            TfidfVectorizer().transform(["x"])

    def test_empty_document_gives_zero_row(self):
        matrix = TfidfVectorizer().fit(DOCS + [""]).transform(DOCS + [""])
        assert sum(matrix[-1]) == 0.0


class TestNgrams:
    def test_word_bigrams(self):
        result = ngrams(["a", "b", "c"], 2)
        assert result == [("a", "b"), ("b", "c")]

    def test_unigrams(self):
        assert ngrams(["a"], 1) == [("a",)]

    def test_n_larger_than_tokens_is_empty(self):
        assert ngrams(["a", "b"], 5) == []

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            ngrams(["a"], 0)

    def test_count_of_trigrams(self):
        tokens = ["w%d" % i for i in range(6)]
        assert len(ngrams(tokens, 3)) == 4

    def test_bigram_features_alias(self):
        assert bigram_features(["x", "y", "z"]) == [("x", "y"), ("y", "z")]


class TestCharNgrams:
    def test_basic_trigrams_with_padding(self):
        assert char_ngrams("abc", n=3) == ["__a", "_ab", "abc", "bc_", "c__"]

    def test_padding_adds_boundary_grams(self):
        result = char_ngrams("ab", n=3)
        assert "__a" in result and "ab_" in result and "b__" in result

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            char_ngrams("word", n=0)
