"""Document similarity: cosine similarity and helpers built on top of it."""

from __future__ import annotations

import math
from typing import Sequence


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors, in [0, 1] for count vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError("vectors must have the same length")
    if not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
