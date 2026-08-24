"""Smoke tests: every example script must run its main() without errors."""

import contextlib
import io

import pytest

from helpers import load_all_examples


def test_all_examples_run(capsys):
    examples = load_all_examples()
    expected = {
        "01_text_preprocessing", "02_stemming", "03_edit_distance",
        "04_features_bow", "05_tfidf", "06_ngrams", "07_similarity",
        "08_naive_bayes_spam", "09_keywords", "10_markov_text_generator",
        "11_sklearn_sentiment",
    }
    assert expected <= set(examples)

    for stem, module in sorted(examples.items()):
        if hasattr(module, "main"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                module.main()
            out = capsys.readouterr().out or buf.getvalue()
            assert len(out.strip()) > 0, f"{stem} produced no output"


def test_example_count():
    assert len(load_all_examples()) >= 11
