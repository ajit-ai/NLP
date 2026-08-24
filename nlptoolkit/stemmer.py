"""A compact rule-based (Porter-style) suffix-stripping stemmer.

This is a simplified, deterministic implementation intended for teaching:
it strips common English inflectional and derivational suffixes step by
step, guarded by a Porter-style "measure" (number of vowel->consonant
transitions) so that stems are never over-truncated.  It is not the full
Porter algorithm, but covers the frequent cases.
"""

from __future__ import annotations

from typing import List

_VOWELS = frozenset("aeiou")


def _contains_vowel(word: str) -> bool:
    return any(c in _VOWELS for c in word)


def _cv_form(word: str) -> str:
    """Return the C/V pattern of a word, e.g. 'tree' -> 'cvv'."""
    return "".join("v" if c in _VOWELS else "c" for c in word)


def _measure(word: str) -> int:
    """Porter measure: number of vowel->consonant transitions."""
    return _cv_form(word).count("vc")


def _ends_double_consonant(word: str) -> bool:
    return (
        len(word) >= 2
        and word[-1] == word[-2]
        and word[-1] not in _VOWELS
    )


def _ends_cvc(word: str) -> bool:
    """True if the word ends consonant-vowel-consonant (not w/x/y)."""
    return (
        len(word) >= 3
        and word[-1] not in _VOWELS | {"w", "x", "y"}
        and word[-2] in _VOWELS
        and word[-3] not in _VOWELS
    )


_STEP1_PLURALS = [
    ("sses", "ss"),
    ("ies", "i"),
    ("ss", "ss"),
    ("s", ""),
]

_STEP2 = [
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("entli", "ent"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
]

_STEP3 = [
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
]

# longest-first so the strongest suffix match wins
_STEP4 = sorted(
    [
        "al", "ance", "ence", "er", "ic", "able", "ible",
        "ant", "ement", "ment", "ent", "ion", "ou", "ism",
        "ate", "iti", "ous", "ive", "ize",
    ],
    key=len,
    reverse=True,
)


def stem(word: str) -> str:
    """Stem a single word using simplified Porter-style rules."""
    if not isinstance(word, str):
        raise TypeError(f"expected str, got {type(word).__name__}")
    w = word.lower().strip()
    if len(w) <= 2 or not w.isalpha():
        return w

    # -- Step 1a: plurals ---------------------------------------------------
    for suffix, repl in _STEP1_PLURALS:
        if w.endswith(suffix):
            w = w[: -len(suffix)] + repl
            break

    # -- Step 1b: -ed / -ing -------------------------------------------------
    if w.endswith("ed") and _contains_vowel(w[:-2]):
        w = w[:-2]
    elif w.endswith("ing") and len(w) > 3 and _contains_vowel(w[:-3]):
        w = w[:-3]
        # undo consonant doubling caused by -ing (running -> runn -> run)
        if _ends_double_consonant(w) and w[-1] not in {"l", "s", "z"}:
            w = w[:-1]

    # -- Step 2: derivational suffixes (stem must contain a vowel) -----------
    for suffix, repl in _STEP2:
        if w.endswith(suffix):
            prefix = w[: -len(suffix)]
            if prefix and _contains_vowel(prefix):
                w = prefix + repl
            break

    # -- Step 3: -ful / -ness / -ical etc. ------------------------------------
    for suffix, repl in _STEP3:
        if w.endswith(suffix):
            prefix = w[: -len(suffix)]
            if prefix and _contains_vowel(prefix):
                w = prefix + repl
            break

    # -- Step 4: residual endings, only if measure(stem) > 1 ------------------
    for suffix in _STEP4:
        if w.endswith(suffix):
            prefix = w[: -len(suffix)]
            if suffix == "ion":
                if prefix.endswith(("s", "t")) and _measure(prefix) > 1:
                    w = prefix
            elif len(prefix) > 1 and _measure(prefix) > 1:
                w = prefix
            break

    # -- Step 5a: drop trailing 'e' when the stem remains pronounceable ------
    if w.endswith("e"):
        without_e = w[:-1]
        m = _measure(without_e)
        if m > 1 or (m == 1 and not _ends_cvc(without_e)):
            w = without_e

    # -- Step 5b: collapse doubled final consonants (except l, s, z) ----------
    if _ends_double_consonant(w) and w[-1] not in {"l", "s", "z"}:
        w = w[:-1]

    return w


def stem_tokens(tokens: List[str]) -> List[str]:
    """Stem an entire token list."""
    return [stem(t) for t in tokens]
