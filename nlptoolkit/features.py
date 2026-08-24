"""Feature extraction for text: one-hot encoding, bag-of-words, TF-IDF,
and character/word n-grams.

Everything is implemented from scratch with only the standard library so
learners can inspect every step.  The example scripts additionally compare
results against scikit-learn.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple


class OneHotEncoder:
    """Map each vocabulary word to a unit vector of length ``len(vocab)``."""

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}

    def fit(self, documents: Iterable[str]) -> "OneHotEncoder":
        index = 0
        for doc in documents:
            for token in doc.split():
                if token not in self.vocab:
                    self.vocab[token] = index
                    index += 1
        return self

    def encode(self, word: str) -> List[float]:
        if word not in self.vocab:
            raise KeyError(f"unknown word: {word!r}")
        vector = [0.0] * len(self.vocab)
        vector[self.vocab[word]] = 1.0
        return vector


class CountVectorizer:
    """Bag-of-words counts over a fitted vocabulary."""

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase
        self.vocabulary_: Dict[str, int] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def _prep(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def fit(self, documents: Sequence[str]) -> "CountVectorizer":
        counter: Counter = Counter()
        for doc in documents:
            counter.update(self._tokenize(doc))
        self.vocabulary_ = {w: i for i, w in enumerate(sorted(counter))}
        return self

    def transform(self, documents: Sequence[str]) -> List[List[int]]:
        vectors = []
        for doc in documents:
            counts = [0] * len(self.vocabulary_)
            for token in self._tokenize(doc):
                idx = self.vocabulary_.get(token)
                if idx is not None:
                    counts[idx] += 1
            vectors.append(counts)
        return vectors

    def fit_transform(self, documents: Sequence[str]) -> List[List[int]]:
        return self.fit(documents).transform(documents)


class TfidfVectorizer:
    """TF-IDF weighting: tf(t, d) * log(N / df(t)) with smoothing."""

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def fit(self, documents: Sequence[str]) -> "TfidfVectorizer":
        n_docs = len(documents)
        assert n_docs > 0, "need at least one document"
        document_frequency: Counter = Counter()
        vocab: set = set()
        for doc in documents:
            tokens = set(self._tokenize(doc))
            document_frequency.update(tokens)
            vocab |= tokens
        self.vocabulary_ = {w: i for i, w in enumerate(sorted(vocab))}
        # smoothed idf avoids division by zero and dampens very common terms
        self.idf_ = {
            term: math.log((1 + n_docs) / (1 + df)) + 1.0
            for term, df in document_frequency.items()
        }
        # unseen-in-fit terms still get a default weight
        self._default_idf = math.log(1 + n_docs) + 1.0
        return self

    def transform(self, documents: Sequence[str]) -> List[List[float]]:
        if not self.vocabulary_:
            raise RuntimeError("fit() must be called before transform()")
        tfidf_matrix = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tf = Counter(tokens)
            row = [0.0] * len(self.vocabulary_)
            norm = 0.0
            for token, count in tf.items():
                idx = self.vocabulary_.get(token)
                if idx is None:
                    continue
                value = (count / len(tokens)) * self.idf_.get(token, self._default_idf)
                row[idx] = value
                norm += value * value
            # L2-normalize so cosine similarity works directly on rows
            if norm > 0:
                norm = math.sqrt(norm)
                row = [v / norm for v in row]
            tfidf_matrix.append(row)
        return tfidf_matrix

    def fit_transform(self, documents: Sequence[str]) -> List[List[float]]:
        return self.fit(documents).transform(documents)


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Generate all word-level n-grams from a token sequence."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def char_ngrams(text: str, n: int = 3, pad: str = "_") -> List[str]:
    """Character n-grams with optional padding at the boundaries."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    padded = pad * (n - 1) + text + pad * (n - 1)
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def bigram_features(tokens: Sequence[str]) -> List[Tuple[str, ...]]:
    """Convenience alias for word bigrams."""
    return ngrams(tokens, 2)
