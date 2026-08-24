"""Tests for Example 03 - edit distance, Jaccard and spell correction."""

import pytest

from nlptoolkit.distance import (
    jaccard_similarity,
    levenshtein_distance,
    normalized_levenshtein,
    spell_correct,
)


class TestLevenshtein:
    @pytest.mark.parametrize("a,b,d", [
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("same", "same", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
        ("intention", "execution", 5),
    ])
    def test_known_distances(self, a, b, d):
        assert levenshtein_distance(a, b) == d

    def test_symmetry(self):
        assert levenshtein_distance("abc", "y") == levenshtein_distance("y", "abc")

    def test_triangle_inequality(self):
        ab = levenshtein_distance("kitten", "smitten")
        bc = levenshtein_distance("smitten", "sitting")
        ac = levenshtein_distance("kitten", "sitting")
        assert ac <= ab + bc

    def test_rejects_non_string(self):
        with pytest.raises(TypeError):
            levenshtein_distance(1, "a")


class TestNormalizedLevenshtein:
    def test_identical_is_zero(self):
        assert normalized_levenshtein("hello", "hello") == 0.0

    def test_completely_different_is_one(self):
        assert normalized_levenshtein("", "abcd") == 1.0

    def test_bounded_between_zero_and_one(self):
        value = normalized_levenshtein("kitten", "sitting")
        assert 0.0 < value <= 1.0

    def test_both_empty(self):
        assert normalized_levenshtein("", "") == 0.0


class TestJaccard:
    def test_identical_sets(self):
        assert jaccard_similarity({1, 2}, {1, 2}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard_similarity({1}, {2}) == 0.0

    def test_partial_overlap(self):
        # |{a,b} ∩ {b,c}| = 1, union = 3
        assert jaccard_similarity({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_both_empty(self):
        assert jaccard_similarity([], []) == 1.0


class TestSpellCorrect:
    VOCAB = ["the", "their", "there", "product", "quality", "great"]

    def test_corrects_single_edit(self):
        assert spell_correct("teh", self.VOCAB) == "the"

    def test_exact_match_returned(self):
        assert spell_correct("quality", self.VOCAB) == "quality"

    def test_unknown_word_outside_max_distance_kept(self):
        assert spell_correct("zzzzzzzz", self.VOCAB, max_distance=2) == "zzzzzzzz"

    def test_deterministic_choice_among_ties(self):
        result = spell_correct("ther", self.VOCAB)
        assert result in {"the", "their", "there"}

    def test_case_insensitive(self):
        assert spell_correct("QUALITY", self.VOCAB) == "quality"
