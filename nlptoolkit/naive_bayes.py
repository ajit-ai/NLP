"""Multinomial Naive Bayes text classifier, implemented from scratch.

Uses Laplace (add-one) smoothing and works on tokenized documents.
Classic use cases: spam filtering and sentiment classification.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List


class NaiveBayesClassifier:
    """Trainable bag-of-words Naive Bayes with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.classes_: List[str] = []
        self.class_log_prior_: Dict[str, float] = {}
        self.word_counts_: Dict[str, Counter] = {}
        self.class_totals_: Dict[str, int] = {}
        self.vocabulary_ = set()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def fit(self, documents: List[str], labels: List[str]) -> "NaiveBayesClassifier":
        if len(documents) != len(labels):
            raise ValueError("documents and labels must have the same length")
        if not documents:
            raise ValueError("cannot fit on an empty dataset")

        self.classes_ = sorted(set(labels))
        self.word_counts_ = {c: Counter() for c in self.classes_}
        self.class_totals_ = {c: 0 for c in self.classes_}
        docs_per_class = Counter(labels)

        for doc, label in zip(documents, labels):
            tokens = self._tokenize(doc)
            self.word_counts_[label].update(tokens)
            self.class_totals_[label] += len(tokens)
            self.vocabulary_.update(tokens)

        n_docs = len(documents)
        vocab_size = len(self.vocabulary_)
        self._vocab_size = vocab_size

        self.class_log_prior_ = {
            c: math.log(docs_per_class[c] / n_docs) for c in self.classes_
        }
        return self

    def predict_log_proba(self, text: str) -> Dict[str, float]:
        """Return log P(class | text), unnormalized but monotonic."""
        tokens = self._tokenize(text)
        scores = {}
        for c in self.classes_:
            score = self.class_log_prior_[c]
            total = self.class_totals_[c] + self.alpha * self._vocab_size
            counts = self.word_counts_[c]
            for token in tokens:
                if token not in self.vocabulary_:
                    continue  # unseen words are ignored at prediction time
                score += math.log((counts[token] + self.alpha) / total)
            scores[c] = score
        return scores

    def predict(self, text: str) -> str:
        scores = self.predict_log_proba(text)
        return max(scores, key=scores.get)

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Normalized posterior probabilities via the log-sum-exp trick."""
        scores = self.predict_log_proba(text)
        peak = max(scores.values())
        exp_shifted = {c: math.exp(s - peak) for c, s in scores.items()}
        total = sum(exp_shifted.values())
        return {c: v / total for c, v in exp_shifted.items()}
