"""Example 03 - Levenshtein edit distance and spelling correction.

Run:  python examples/03_edit_distance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlptoolkit.distance import (
    jaccard_similarity,
    levenshtein_distance,
    normalized_levenshtein,
    spell_correct,
)

VOCAB = ["the", "their", "there", "these", "product", "products", "quality",
         "great", "grate", "hello", "help", "world", "would", "cat", "cart"]


def main() -> None:
    pairs = [("kitten", "sitting"), ("flaw", "lawn"), ("", "abc"), ("same", "same")]
    print("Levenshtein distances:")
    for a, b in pairs:
        print(f"  {a!r} vs {b!r} -> {levenshtein_distance(a, b)}")

    print("\nNormalized distances (0 = identical):")
    for a, b in pairs:
        print(f"  {a!r} vs {b!r} -> {normalized_levenshtein(a, b):.3f}")

    print("\nJaccard similarity of token sets:")
    doc_a = "the product quality is great".split()
    doc_b = "the product quality is really great".split()
    print(f"  {jaccard_similarity(doc_a, doc_b):.3f}")

    print("\nSpelling correction against a small vocabulary:")
    for typo in ["teh", "prodact", "qualty", "gret", "wrold"]:
        corrected = spell_correct(typo, VOCAB)
        print(f"  {typo:<10} -> {corrected}")


if __name__ == "__main__":
    main()
