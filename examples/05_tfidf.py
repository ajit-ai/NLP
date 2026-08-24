"""Example 05 - TF-IDF weighting, from scratch and with scikit-learn.

Run:  python examples/05_tfidf.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf
except ImportError:  # pragma: no cover - sklearn optional for this demo
    SkTfidf = None

from nlptoolkit.features import TfidfVectorizer

DOCS = [
    "the cat sat on the mat",
    "the dog barked at the cat",
    "machine learning models learn patterns from data",
    "deep learning is a branch of machine learning",
]


def top_terms(vectorizer, matrix, doc_index: int, k: int = 3):
    vocab = sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get)
    row = matrix[doc_index]
    pairs = sorted(zip(vocab, row), key=lambda p: -p[1])
    return [(term, round(float(score), 4)) for term, score in pairs[:k] if score > 0]


def main() -> None:
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(DOCS)

    print("TF-IDF (nlptoolkit, L2-normalized rows):")
    for i, doc in enumerate(DOCS):
        print(f"  doc {i}: {top_terms(vectorizer, matrix, i)}")

    if SkTfidf is not None:
        sk_vec = SkTfidf()
        sk_matrix = sk_vec.fit_transform(DOCS).toarray()

        ours = np.array(matrix)
        theirs = np.array(sk_matrix)
        # compare the dominant term per document rather than raw values,
        # because smoothing details differ between implementations
        def argmax_terms(m, vocab):
            vocab_sorted = sorted(vocab, key=vocab.get)
            return [vocab_sorted[int(np.argmax(row))] for row in m]

        our_top = argmax_terms(ours, vectorizer.vocabulary_)
        sk_top = argmax_terms(theirs, sk_vec.vocabulary_)
        print("\nMost important term per document:")
        print("  nlptoolkit  :", our_top)
        print("  scikit-learn:", sk_top)
        assert our_top == sk_top, "implementations disagree!"
        print("  -> implementations agree on the dominant term of every doc")


if __name__ == "__main__":
    main()
