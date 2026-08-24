"""Example 04 - Feature extraction: one-hot encoding and bag-of-words.

Run:  python examples/04_features_bow.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlptoolkit.features import CountVectorizer, OneHotEncoder

DOCS = [
    "the cat sat on the mat",
    "the dog ate my homework",
    "cats and dogs are friends",
]


def main() -> None:
    print("Documents:")
    for doc in DOCS:
        print("  -", doc)

    # ---- One-hot encoding -------------------------------------------------
    encoder = OneHotEncoder().fit(DOCS)
    print("\nOne-hot vocabulary:", encoder.vocab)
    for word in ["cat", "dog", "mat"]:
        print(f"  one-hot({word!r}) = {encoder.encode(word)}")

    # ---- Bag of words ------------------------------------------------------
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(DOCS)
    vocab = sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get)
    print("\nBoW vocabulary:", vocab)
    print("Document-term matrix:")
    header = "doc | " + " ".join(f"{w:>7}" for w in vocab)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(matrix):
        print(f" {i}  | " + " ".join(f"{v:>7}" for v in row))

    print("\nTransform unseen document (unknown words ignored):")
    print(" ", vectorizer.transform(["the cat and the unknown zebra"]))


if __name__ == "__main__":
    main()
