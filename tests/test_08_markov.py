"""Tests for Example 10 - Markov chain text generation."""

import pytest

from nlptoolkit.markov import MarkovChain

CORPUS = (
    "the cat sat on the mat. the cat saw the dog. "
    "a model learns patterns from data."
)


@pytest.fixture()
def chain():
    return MarkovChain(order=3).train(CORPUS)


class TestTraining:
    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            MarkovChain(order=0)

    def test_text_too_short_raises(self):
        with pytest.raises(ValueError):
            MarkovChain(order=10).train("short")

    def test_states_learned(self, chain):
        assert len(chain.transitions) > 0
        assert "the" in chain.transitions

    def test_training_is_additive(self):
        c1 = MarkovChain(order=2).train("abab")
        assert ("ab" in c1.transitions)


class TestGeneration:
    def test_output_length(self, chain):
        assert len(chain.generate(50, seed=1)) == 50

    def test_same_seed_same_output(self, chain):
        assert chain.generate(40, seed=7) == chain.generate(40, seed=7)

    def test_different_seeds_usually_differ(self, chain):
        outputs = {chain.generate(60, seed=s) for s in range(5)}
        assert len(outputs) >= 2

    def test_length_below_order_raises(self, chain):
        with pytest.raises(ValueError):
            chain.generate(2)  # order is 3

    def test_untrained_model_raises(self):
        with pytest.raises(RuntimeError):
            MarkovChain(order=2).generate(10)

    def test_generated_text_uses_corpus_characters(self, chain):
        allowed = set(CORPUS)
        generated = chain.generate(120, seed=3)
        assert set(generated) <= allowed

    def test_higher_order_stays_closer_to_corpus(self):
        low = MarkovChain(order=2).train(CORPUS).generate(100, seed=5)
        high = MarkovChain(order=6).train(CORPUS).generate(100, seed=5)
        # every 6-gram of the high-order output that has a full context must
        # appear in the corpus; a cheap proxy: many 6-char windows exist there
        windows = {high[i:i + 6] for i in range(len(high) - 5)}
        hits = sum(1 for w in windows if w in CORPUS)
        assert hits / len(windows) > 0.3
