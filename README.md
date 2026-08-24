# NLP

A collection of Natural Language Processing (NLP) examples, feature
engineering techniques and classic algorithms — each implemented from
scratch (stdlib only) **plus** runnable example scripts, **plus** a pytest
test suite covering every program.

## Repository layout

```
nlptoolkit/          Core library (pure Python implementations)
├── preprocessing    cleaning, tokenization, normalization, stopwords
├── stemmer          rule-based Porter-style stemming
├── distance         Levenshtein edit distance + spell correction
├── features         one-hot, bag-of-words, TF-IDF, n-grams
├── similarity       cosine similarity
├── naive_bayes      multinomial NB text classifier (from scratch)
├── markov           character-level Markov-chain language model
└── keywords         frequency-based & RAKE-style keyword extraction

examples/            Runnable demo scripts (one per topic)
tests/               Pytest suite - at least one file per program
context/             Amazon product reviews dataset (CSV)
```

## Examples

| #  | Script                              | Topic                                              |
|----|-------------------------------------|----------------------------------------------------|
| 01 | `examples/01_text_preprocessing.py` | Cleaning, word/sentence tokenization, stopwords     |
| 02 | `examples/02_stemming.py`           | Rule-based (Porter-style) stemming                  |
| 03 | `examples/03_edit_distance.py`      | Levenshtein distance, Jaccard, spell correction     |
| 04 | `examples/04_features_bow.py`       | One-hot encoding & bag-of-words features            |
| 05 | `examples/05_tfidf.py`              | TF-IDF from scratch vs scikit-learn                 |
| 06 | `examples/06_ngrams.py`             | Word n-grams and character n-grams                  |
| 07 | `examples/07_similarity.py`         | Cosine document similarity                          |
| 08 | `examples/08_naive_bayes_spam.py`   | Spam classification with from-scratch Naive Bayes   |
| 09 | `examples/09_keywords.py`           | Frequency & RAKE keyword extraction                 |
| 10 | `examples/10_markov_text_generator.py` | Character-level Markov language model            |
| 11 | `examples/11_sklearn_sentiment.py`  | Sentiment analysis with sklearn pipelines           |
| 12 | `examples/12_amazon_reviews_sentiment.py` | Real Amazon reviews dataset end-to-end        |

Run any example:

```bash
python examples/01_text_preprocessing.py
```

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
pytest                                            # run the whole suite
pytest --cov=nlptoolkit                           # with coverage
```

The core library (`nlptoolkit`) uses only the Python standard library;
`numpy`, `pandas` and `scikit-learn` are needed for examples 05, 11 and 12.

## Notebook

`Amazon_Product_Reviews-Sentiments.ipynb` performs sentiment analysis on the
Amazon product reviews dataset in `context/`.

## Branching model

- `main` — stable branch
- `develop` — integration branch; feature work lands here first and is
  merged into `main`
