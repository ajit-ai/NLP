"""String distance metrics and a tiny spelling corrector.

Includes the classic dynamic-programming Levenshtein (edit) distance with
insertion, deletion and substitution costs of 1, plus Jaccard set
similarity over token sets.
"""

from __future__ import annotations

from typing import Dict, Iterable, List


def levenshtein_distance(a: str, b: str) -> int:
    """Minimum number of single-character edits to turn ``a`` into ``b``."""
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError("levenshtein_distance expects two strings")
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, substitute_cost))
        previous = current
    return previous[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    """Edit distance scaled to [0, 1] where 0 means identical strings."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return levenshtein_distance(a, b) / longest


def jaccard_similarity(a: Iterable, b: Iterable) -> float:
    """Jaccard similarity |A ∩ B| / |A ∪ B| between two iterables."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union)


def spell_correct(word: str, vocabulary: Iterable[str], max_distance: int = 2) -> str:
    """Return the closest vocabulary word within ``max_distance`` edits.

    Falls back to the original word when nothing is close enough.
    """
    candidates: Dict[int, List[str]] = {}
    for candidate in sorted(set(vocabulary)):
        d = levenshtein_distance(word.lower(), candidate.lower())
        if d <= max_distance:
            candidates.setdefault(d, []).append(candidate)
    if not candidates or 0 in candidates:
        return word if not candidates else candidates[0][0]
    # deterministic pick: alphabetically first among the nearest words
    nearest = min(candidates)
    return sorted(candidates[nearest])[0]
