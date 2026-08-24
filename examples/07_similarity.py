"""Example 07 - Document similarity with cosine and TF-IDF.

Run:  python examples/07_similarity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlptoolkit.distance import jaccard_similarity
from nlptoolkit.features import CountVectorizer
from nlptoolkit.similarity import cosine_similarity

DOCS = [
    "the cat sat on the mat",
    "the cat played on the mat",
    "python is a great programming language",
    "java is also a programming language",
]


def main() -> None:
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(DOCS)

    print("Pairwise cosine similarity (bag-of-words vectors):")
    n = len(DOCS)
    header = "     " + "".join(f"doc{j:>3}" for j in range(n))
    print(header)
    for i in range(n):
        row = f"doc{i:<3}"
        for j in range(n):
            sim = cosine_similarity(matrix[i], matrix[j])
            row += f"{sim:>6.2f}"
        print(row)

    print("\nInterpretation:")
    print(f"  doc0 vs doc1 -> {cosine_similarity(matrix[0], matrix[1]):.2f}  (same topic, near-duplicates)")
    print(f"  doc0 vs doc2 -> {cosine_similarity(matrix[0], matrix[2]):.2f}  (unrelated topics)")

    print("\nJaccard token-set similarity for the same pairs:")
    t0, t2 = set(DOCS[0].split()), set(DOCS[2].split())
    print(f"  doc0 vs doc2 -> {jaccard_similarity(t0, t2):.2f}")


if __name__ == "__main__":
    main()
