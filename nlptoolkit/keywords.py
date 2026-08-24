"""Lightweight keyword extraction.

Two complementary strategies:
1. Frequency score with stopword filtering and position boost.
2. RAKE-style phrase scoring: candidate phrases are split on stopwords,
   and a phrase's score is the sum of its words' degree/frequency ratio.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from .preprocessing import DEFAULT_STOPWORDS


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def frequency_keywords(
    text: str,
    top_k: int = 5,
    stopwords=DEFAULT_STOPWORDS,
) -> List[Tuple[str, float]]:
    """Top-k content words ranked by normalized frequency.

    Words appearing earlier get a small linear boost, mimicking the
    intuition that leading sentences carry topic keywords.
    """
    tokens = _words(text)
    if not tokens:
        return []
    stop = set(stopwords)
    scored = Counter()
    total = 0
    for i, token in enumerate(tokens):
        if token in stop:
            continue
        # earlier words get a small boost (leading sentences carry topics)
        position_boost = 1.0 + (1.0 - i / len(tokens)) * 0.5
        scored[token] += position_boost
        total += 1
    if not total:
        return []
    max_score = max(scored.values()) or 1.0
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(w, round(score / max_score, 4)) for w, score in ranked[:top_k]]


def rake_keywords(
    text: str,
    top_k: int = 5,
    stopwords=DEFAULT_STOPWORDS,
) -> List[Tuple[str, float]]:
    """RAKE-lite: score phrases by sum of word degree/frequency ratios."""
    stop = set(stopwords)
    tokens = _words(text)
    # split into candidate phrases at stopwords
    phrases: List[List[str]] = [[]]
    for token in tokens:
        if token in stop:
            if phrases[-1]:
                phrases.append([])
        else:
            phrases[-1].append(token)
    phrases = [p for p in phrases if p]

    freq: Counter = Counter()
    degree: Counter = Counter()
    for phrase in phrases:
        for word in phrase:
            freq[word] += 1
            degree[word] += len(phrase) - 1

    phrase_scores: List[Tuple[str, float]] = []
    for phrase in phrases:
        score = 0.0
        for word in phrase:
            f = freq[word] or 1
            d = degree[word]
            score += d / f
        phrase_scores.append((" ".join(phrase), score))

    phrase_scores.sort(key=lambda kv: (-kv[1], kv[0]))
    return phrase_scores[:top_k]


def keyword_summary(text: str, top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
    """Run both extractors and return their results side by side."""
    return {
        "frequency": frequency_keywords(text, top_k=top_k),
        "rake": rake_keywords(text, top_k=top_k),
    }
