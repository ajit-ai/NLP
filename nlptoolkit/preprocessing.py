"""Text preprocessing utilities: cleaning, tokenization, normalization.

All functions are pure and dependency-free so they are easy to test.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import List

DEFAULT_STOPWORDS = frozenset(
    """a an the and or but if then else when while of to in on at by for with
    about against between into through during before after above below from up
    down out off over under again further once here there all any both each few
    more most other some such no nor not only own same so than too very can will
    just don should now is am are was were be been being have has had having do
    does did doing it its this that these those i me my we our you your he him
    his she her they them their what which who whom how why as until because
    s t d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma
    mightn mustn needn shan shouldn wasn weren won wouldn""".split()
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_MENTION_RE = re.compile(r"[@#]\w+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_html: bool = True,
    remove_mentions: bool = True,
) -> str:
    """Normalize raw text: strip URLs/HTML/mentions, collapse whitespace."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    if remove_urls:
        text = _URL_RE.sub(" ", text)
    if remove_html:
        text = _HTML_RE.sub(" ", text)
    if remove_mentions:
        text = _MENTION_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def word_tokenize(text: str, keep_punctuation: bool = False) -> List[str]:
    """Split text into word tokens.

    Punctuation is stripped by default; with ``keep_punctuation=True``
    punctuation characters become standalone tokens.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    tokens = []
    for raw in text.split():
        stripped = raw.strip(string.punctuation)
        if keep_punctuation:
            tokens.append(raw)
        elif stripped:
            tokens.append(stripped)
    return [t for t in tokens if t]


def sentence_tokenize(text: str) -> List[str]:
    """Split text into sentences on ``.``, ``!`` and ``?`` boundaries."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def normalize_word(word: str, lowercase: bool = True) -> str:
    """Lowercase + strip surrounding punctuation + NFKC-normalize one word."""
    word = unicodedata.normalize("NFKC", word)
    if lowercase:
        word = word.lower()
    return word.strip(string.punctuation)


def remove_stopwords(tokens: List[str], stopwords=DEFAULT_STOPWORDS) -> List[str]:
    """Filter a token list, dropping every token found in ``stopwords``."""
    stop = set(stopwords)
    return [t for t in tokens if t.lower() not in stop]


def preprocess(
    text: str,
    remove_stops: bool = False,
    stemmer=None,
) -> List[str]:
    """Full pipeline: clean -> tokenize -> (optional) stopword/stem filter."""
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    if stemmer is not None:
        tokens = [stemmer(t) for t in tokens]
    return tokens
