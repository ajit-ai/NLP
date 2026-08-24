"""Example 01 - Text preprocessing pipeline.

Demonstrates cleaning, word/sentence tokenization, normalization and
stopword removal.

Run:  python examples/01_text_preprocessing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlptoolkit.preprocessing import (
    clean_text,
    preprocess,
    remove_stopwords,
    sentence_tokenize,
    word_tokenize,
)

RAW_TEXT = (
    "<p>Check out https://example.com/deal!!! @alice said: "
    "The NEW Product is AMAZING. It arrived in 2 days!</p> "
    "But the packaging was terrible... Would you buy it again?"
)


def main() -> None:
    print("Raw text:")
    print(" ", RAW_TEXT)

    cleaned = clean_text(RAW_TEXT)
    print("\n1) Cleaned text (URLs/HTML/mentions removed):")
    print(" ", cleaned)

    print("\n2) Word tokens:")
    print(" ", word_tokenize(cleaned))

    print("\n3) Sentences:")
    for sentence in sentence_tokenize(cleaned):
        print("  -", sentence)

    tokens = word_tokenize(cleaned)
    print("\n4) Tokens without stopwords:")
    print(" ", remove_stopwords(tokens))

    print("\n5) Full pipeline (clean + tokenize + stopword filter):")
    print(" ", preprocess(RAW_TEXT, remove_stops=True))


if __name__ == "__main__":
    main()
