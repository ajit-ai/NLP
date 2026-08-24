"""Tests for Example 02 - the rule-based stemmer."""

import pytest

from nlptoolkit.stemmer import stem, stem_tokens


class TestPluralRules:
    @pytest.mark.parametrize("word,expected", [
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("cats", "cat"),
    ])
    def test_step1_plurals(self, word, expected):
        assert stem(word) == expected

    def test_singular_unchanged(self):
        assert stem("cat") == "cat"


class TestInflectionRules:
    @pytest.mark.parametrize("word,expected", [
        ("running", "run"),
        ("connected", "connect"),
        ("connections", "connect"),
        ("abilities", "abil"),
    ])
    def test_ed_ing_plural_forms(self, word, expected):
        assert stem(word) == expected


class TestDerivationalRules:
    @pytest.mark.parametrize("word,expected", [
        ("relational", "relat"),
        ("conditional", "condit"),
        ("rational", "ration"),
        ("nationalization", "nation"),
        ("connection", "connect"),
    ])
    def test_step2(self, word, expected):
        assert stem(word) == expected


class TestCleanupRules:
    def test_beautiful(self):
        assert stem("beautiful") == "beauti"

    def test_happily_keeps_root(self):
        assert stem("happily").startswith("happ")

    def test_measure_guard_prevents_overstemming(self):
        # 'relate' must not be truncated to 'rel'
        assert stem("relate").startswith("relat")
        assert stem("rate") == "rate"


class TestEdgeCases:
    def test_short_words_returned_as_is(self):
        assert stem("a") == "a"
        assert stem("is") == "is"

    def test_non_alpha_passthrough(self):
        assert stem("123!") == "123!"

    def test_uppercase_input_lowered(self):
        assert stem("Cats") == stem("cats")

    def test_empty_string(self):
        assert stem("") == ""

    def test_rejects_non_string(self):
        with pytest.raises(TypeError):
            stem(42)

    def test_idempotent(self):
        once = stem("connections")
        assert stem(once) == once

    def test_morphological_family_collapses(self):
        family = ["connect", "connected", "connecting", "connections"]
        stems = {stem(w) for w in family}
        assert stems == {"connect"}


def test_stem_tokens_maps_every_element():
    result = stem_tokens(["cats", "running", "abilities"])
    assert result == ["cat", "run", "abil"]
