"""Example 08 - Spam classification with a from-scratch Naive Bayes.

Run:  python examples/08_naive_bayes_spam.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlptoolkit.naive_bayes import NaiveBayesClassifier

TRAIN_DOCS = [
    "win a free iphone now click this link",
    "cheap meds online buy now discount",
    "congratulations you won a lottery prize claim now",
    "limited offer free money transfer today",
    "hey can we reschedule the meeting to tomorrow",
    "please review the attached project report",
    "lunch tomorrow at the usual place",
    "here are the notes from today's meeting",
]
TRAIN_LABELS = ["spam", "spam", "spam", "spam", "ham", "ham", "ham", "ham"]

TEST_DOCS = [
    "free prize winner click now",
    "meeting notes from yesterday",
    "buy cheap lottery tickets online",
]


def main() -> None:
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(TRAIN_DOCS, TRAIN_LABELS)

    print("Classes:", clf.classes_)
    print("\nPredictions with class probabilities:")
    for doc in TEST_DOCS:
        label = clf.predict(doc)
        proba = clf.predict_proba(doc)
        print(f"  {doc!r}")
        print(f"    -> {label}  "
              f"(spam={proba['spam']:.3f}, ham={proba['ham']:.3f})")

    # sanity check on the training set
    train_preds = [clf.predict(d) for d in TRAIN_DOCS]
    accuracy = sum(p == y for p, y in zip(train_preds, TRAIN_LABELS)) / len(TRAIN_LABELS)
    print(f"\nTraining accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
