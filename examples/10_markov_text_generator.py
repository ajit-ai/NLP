"""Example 10 - Character-level Markov chain text generation.

Trains a tiny language model on a sample corpus and samples new
pseudo-words/sentences with a fixed random seed.

Run:  python examples/10_markov_text_generator.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlptoolkit.markov import MarkovChain

CORPUS = (
    "the cat sat on the mat. the cat saw the dog. the dog chased the cat. "
    "a good model learns patterns from data. a model learns from examples. "
    "language models learn to predict the next character in a sequence."
)


def main() -> None:
    chain = MarkovChain(order=3).train(CORPUS)
    print(f"Corpus length: {len(CORPUS)} chars, states: {len(chain.transitions)}")

    print("\nGenerated text (order 3):")
    for seed in [1, 7, 42]:
        generated = chain.generate(80, seed=seed)
        print(f"  seed={seed}: {generated}")

    higher_order = MarkovChain(order=5).train(CORPUS)
    print("\nGenerated text (order 5, closer to the training corpus):")
    print(" ", higher_order.generate(90, seed=42))

    # determinism check: same seed -> same output
    assert higher_order.generate(90, seed=42) == higher_order.generate(90, seed=42)
    print("\nSame seed reproduces identical output (deterministic).")


if __name__ == "__main__":
    main()
