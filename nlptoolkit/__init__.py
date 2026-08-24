"""nlptoolkit: a small, dependency-light NLP teaching toolkit.

Modules
-------
preprocessing   : cleaning, tokenization, normalization, stopword removal
stemmer         : rule-based suffix-stripping stemmer (Porter-style)
distance        : Levenshtein edit distance + spelling correction
features        : one-hot, bag-of-words, TF-IDF and n-gram feature builders
similarity      : cosine and Jaccard similarity
naive_bayes     : multinomial Naive Bayes text classifier (from scratch)
markov          : Markov-chain language model / text generator
keywords        : frequency- and position-based keyword extraction
"""

from . import (
    preprocessing,
    stemmer,
    distance,
    features,
    similarity,
    naive_bayes,
    markov,
    keywords,
)

__version__ = "0.1.0"

__all__ = [
    "preprocessing",
    "stemmer",
    "distance",
    "features",
    "similarity",
    "naive_bayes",
    "markov",
    "keywords",
]
