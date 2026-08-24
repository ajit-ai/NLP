"""Tests for Example 01 - text preprocessing."""

import pytest

from nlptoolkit.preprocessing import (
    clean_text,
    normalize_word,
    preprocess,
    remove_stopwords,
    sentence_tokenize,
    word_tokenize,
)


class TestCleanText:
    def test_removes_urls(self):
        assert clean_text("visit https://example.com now") == "visit now"

    def test_removes_html_tags(self):
        assert clean_text("<b>hello</b> world") == "hello world"

    def test_removes_mentions_and_hashtags(self):
        assert clean_text("hi @alice check #sale") == "hi check"

    def test_lowercases_by_default(self):
        assert clean_text("Hello WORLD") == "hello world"

    def test_lowercase_disabled(self):
        assert clean_text("Hello WORLD", lowercase=False) == "Hello WORLD"

    def test_collapses_whitespace(self):
        assert clean_text("a\n  b\t c ") == "a b c"

    def test_rejects_non_string(self):
        with pytest.raises(TypeError):
            clean_text(123)


class TestWordTokenize:
    def test_basic_splitting(self):
        assert word_tokenize("the cat sat") == ["the", "cat", "sat"]

    def test_strips_punctuation(self):
        assert word_tokenize("great, product!") == ["great", "product"]

    def test_keep_punctuation_mode(self):
        tokens = word_tokenize("wow!!!", keep_punctuation=True)
        assert tokens == ["wow!!!"]

    def test_empty_string(self):
        assert word_tokenize("") == []

    def test_rejects_non_string(self):
        with pytest.raises(TypeError):
            word_tokenize(None)


class TestSentenceTokenize:
    def test_periods(self):
        assert sentence_tokenize("One. Two. Three.") == ["One.", "Two.", "Three."]

    def test_mixed_punctuation(self):
        result = sentence_tokenize("Really? Yes! Ok.")
        assert result == ["Really?", "Yes!", "Ok."]

    def test_single_sentence(self):
        assert sentence_tokenize("just one here") == ["just one here"]

    def test_abbrev_decimal_not_split(self):
        # decimals like 3.14 should not split
        assert len(sentence_tokenize("Pi is 3.14 ok.")) == 1


class TestRemoveStopwords:
    def test_removes_common_words(self):
        tokens = ["the", "cat", "is", "on", "the", "mat"]
        assert remove_stopwords(tokens) == ["cat", "mat"]

    def test_keeps_content_words(self):
        assert remove_stopwords(["machine", "learning"]) == ["machine", "learning"]

    def test_custom_stopword_list(self):
        assert remove_stopwords(["a", "b"], stopwords={"a"}) == ["b"]


class TestNormalizeWord:
    def test_lowercases_and_strips_punct(self):
        assert normalize_word("Hello!") == "hello"

    def test_keeps_inner_apostrophe(self):
        assert normalize_word("don't.") == "don't"


class TestFullPipeline:
    def test_preprocess_with_stopwords(self):
        tokens = preprocess("The cats are running!", remove_stops=True)
        assert "the" not in tokens
        assert "cats" in tokens

    def test_preprocess_with_custom_stemmer(self):
        tokens = preprocess("The dogs are running", remove_stops=True,
                            stemmer=lambda w: w[:-1] if w.endswith("s") else w)
        assert "dog" in tokens

    def test_deterministic(self):
        text = "Some sample TEXT, with punctuation."
        assert preprocess(text) == preprocess(text)
