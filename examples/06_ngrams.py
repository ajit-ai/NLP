"""Example 06 - N-grams: word bigrams/trigrams and character n-grams.

Run:  python examples/06_ngrams.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlptoolkit.features import char_ngrams, ngrams

SENTENCE = "the quick brown fox jumps over the lazy dog".split()


def main() -> None:
    print("Tokens:", SENTENCE)

    print("\nUnigrams (n=1):", ngrams(SENTENCE, 1)[:5], "...")
    print("\nBigrams (n=2):")
    for bg in ngrams(SENTENCE, 2):
        print("  ", bg)
    print("\nTrigrams (n=3):")
    for tg in ngrams(SENTENCE, 3):
        print("  ", tg)

    # n-gram counts as simple language-model statistics
    from collections import Counter
    bigram_counts = Counter(ngrams(SENTENCE, 2))
    print("\nMost common bigrams:", bigram_counts.most_common(3))

    # character n-grams are useful for spelling and short-text matching
    word = "language"
    print(f"\nCharacter trigrams of {word!r}:")
    print(" ", char_ngrams(word, n=3))


if __name__ == "__main__":
    main()
