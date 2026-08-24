"""Example 02 - Stemming with a rule-based Porter-style stemmer.

Run:  python examples/02_stemming.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlptoolkit.preprocessing import preprocess
from nlptoolkit.stemmer import stem, stem_tokens

WORDS = [
    "caresses", "ponies", "cats", "running", "runs", "happily",
    "relational", "conditional", "rational", "nationalization",
    "connection", "beautiful", "carelessness", "abilities",
]

REVIEW = (
    "The movies were amazing and the actors running scenes were fantastic. "
    "I loved the connections between the characters; the abilities shown "
    "were unbelievable."
)


def main() -> None:
    print("Word -> Stem")
    print("-" * 30)
    for word in WORDS:
        print(f"{word:<18} -> {stem(word)}")

    tokens = preprocess(REVIEW, remove_stops=True)
    stemmed = stem_tokens(tokens)
    print("\nReview tokens (stopwords removed):")
    print(" ", tokens)
    print("Stemmed review:")
    print(" ", stemmed)

    # vocabulary compression demo
    family = ["connect", "connected", "connecting", "connections", "connects"]
    print("\nVocabulary compression:")
    print(" before:", len(set(family)), "unique forms:", sorted(set(family)))
    print(" after :", len(set(stem_tokens(family))), "unique stems:",
          sorted(set(stem_tokens(family))))


if __name__ == "__main__":
    main()
