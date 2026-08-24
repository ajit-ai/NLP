"""A character-level Markov-chain language model for text generation.

Learns transition probabilities from an example corpus and samples new
text.  Supports variable order (n = 1..5) with deterministic seeding so
generated output is reproducible in tests.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List


class MarkovChain:
    """Order-n Markov model over characters (or any token sequence)."""

    def __init__(self, order: int = 3) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.transitions: Dict[str, Counter] = defaultdict(Counter)
        self.starts: List[str] = []

    def train(self, text: str) -> "MarkovChain":
        if len(text) <= self.order:
            raise ValueError("training text too short for this order")
        n = self.order
        for i in range(len(text) - n):
            state = text[i : i + n]
            nxt = text[i + n]
            if i == 0:
                self.starts.append(state)
            self.transitions[state][nxt] += 1
        return self

    def _next_char(self, state: str, rng: random.Random) -> str:
        counter = self.transitions[state]
        if not counter:
            raise KeyError(f"state {state!r} never observed")
        population: List[str] = []
        weights: List[int] = []
        total = 0
        for ch, count in sorted(counter.items()):  # deterministic ordering
            population.append(ch)
            weights.append(count)
            total += count
        pick = rng.choices(population, weights=weights, k=1)[0]
        return pick

    def generate(self, length: int, seed: int | None = None) -> str:
        """Sample ``length`` characters starting from an observed prefix."""
        if length < self.order:
            raise ValueError("length must be >= order")
        rng = random.Random(seed)
        if self.starts:
            state = rng.choice(sorted(self.starts))
        else:
            raise RuntimeError("model is not trained")
        out = [state]
        while len(out[-1]) < length:
            try:
                ch = self._next_char(out[-1][-self.order :], rng)
            except KeyError:
                break  # dead end: stop generation gracefully
            out.append(out[-1] + ch)
        generated = out[-1][:length]

        # fall back to unigram sampling if the chain died early
        while len(generated) < length and self.transitions:
            all_chars: List[str] = []
            all_weights: List[int] = []
            seen: set = set()
            for counter in self.transitions.values():
                for ch, count in sorted(counter.items()):
                    if ch not in seen:
                        all_chars.append(ch)
                        all_weights.append(count)
                        seen.add(ch)
            generated += rng.choices(all_chars, weights=all_weights, k=1)[0]
        return generated
