"""Tests for Example 07 - cosine similarity."""

import math

import pytest

from nlptoolkit.similarity import cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_negative_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_scale_invariant(self):
        a = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, [2.0, 4.0, 6.0]) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_empty_vectors_return_zero(self):
        assert cosine_similarity([], []) == 0.0

    def test_known_angle_45_degrees(self):
        # angle between (1,0) and (1,1) is 45 degrees -> cos = sqrt(2)/2
        assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(math.sqrt(2) / 2)
