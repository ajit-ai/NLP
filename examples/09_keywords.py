"""Example 09 - Keyword extraction (frequency-based and RAKE-style).

Run:  python examples/09_keywords.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlptoolkit.keywords import frequency_keywords, keyword_summary, rake_keywords

TEXT = (
    "Natural language processing enables computers to understand human "
    "language. Deep learning models power modern natural language systems. "
    "Language translation, sentiment analysis and text summarization are "
    "popular language processing applications."
)


def main() -> None:
    print("Text:")
    print(" ", TEXT)

    print("\nFrequency keywords (word, normalized score):")
    for word, score in frequency_keywords(TEXT, top_k=8):
        print(f"  {word:<16} {score:.4f}")

    print("\nRAKE-style phrase keywords:")
    for phrase, score in rake_keywords(TEXT, top_k=8):
        print(f"  {phrase:<32} {score:.3f}")

    summary = keyword_summary(TEXT, top_k=5)
    assert set(summary) == {"frequency", "rake"}
    print("\nTop-5 comparison:", {k: [w for w, _ in v] for k, v in summary.items()})


if __name__ == "__main__":
    main()
