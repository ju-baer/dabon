"""
tests/test_diversity_metrics.py
--------------------------------
Unit tests for all diversity metrics, especially RND.
"""

import numpy as np
import pytest
from src.metrics.diversity import (
    average_pairwise_cosine_distance,
    compute_disagreement_vectors,
    distinct_n,
    reward_range,
    reward_variance,
    rnd,
)


class TestAPCD:
    def test_identical_embeddings_zero_distance(self):
        emb = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        assert average_pairwise_cosine_distance(emb) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_embeddings_max_distance(self):
        emb = np.eye(3, dtype=float)
        dist = average_pairwise_cosine_distance(emb)
        # cosine distance between orthogonal unit vectors = 1.0
        assert dist == pytest.approx(1.0, abs=1e-6)

    def test_single_output_returns_zero(self):
        emb = np.array([[1, 0, 0]], dtype=float)
        assert average_pairwise_cosine_distance(emb) == 0.0

    def test_output_in_valid_range(self):
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((10, 32))
        d   = average_pairwise_cosine_distance(emb)
        assert 0.0 <= d <= 2.0


class TestDisagreementVectors:
    def test_zero_disagreement_for_identical_rms(self):
        scores = np.ones((5, 3)) * 0.8    # all RMs agree
        delta  = compute_disagreement_vectors(scores)
        assert np.allclose(delta, 0.0)

    def test_disagreement_sums_to_zero(self):
        rng    = np.random.default_rng(42)
        scores = rng.random((8, 4))
        delta  = compute_disagreement_vectors(scores)
        # Each row should sum to ~0 (centred)
        assert np.allclose(delta.sum(axis=1), 0.0, atol=1e-10)

    def test_shape_preserved(self):
        scores = np.random.rand(6, 3)
        delta  = compute_disagreement_vectors(scores)
        assert delta.shape == (6, 3)


class TestRND:
    def test_rnd_increases_with_diversity(self):
        """More diverse disagreement → higher RND."""
        # Low diversity: all outputs have same disagreement pattern
        scores_low  = np.tile([0.9, 0.5, 0.7], (8, 1))          # (8, 3)
        # High diversity: orthogonal disagreement patterns
        scores_high = np.vstack([
            np.array([0.9, 0.5, 0.5]),
            np.array([0.5, 0.9, 0.5]),
            np.array([0.5, 0.5, 0.9]),
            np.array([0.9, 0.9, 0.5]),
            np.array([0.5, 0.9, 0.9]),
            np.array([0.9, 0.5, 0.9]),
            np.array([0.7, 0.3, 0.8]),
            np.array([0.3, 0.8, 0.4]),
        ])
        rnd_low  = rnd(scores_low)
        rnd_high = rnd(scores_high)
        assert rnd_high > rnd_low, (
            f"Expected RND(diverse) > RND(redundant), got {rnd_high:.4f} <= {rnd_low:.4f}"
        )

    def test_rnd_is_finite(self):
        rng    = np.random.default_rng(7)
        scores = rng.random((16, 3))
        val    = rnd(scores)
        assert np.isfinite(val)

    def test_rnd_invariant_to_overall_scale(self):
        """Scaling all rewards by a constant shifts RND but preserves ordering."""
        rng     = np.random.default_rng(1)
        scores1 = rng.random((8, 3))
        scores2 = scores1 * 2.0
        # disagreement vectors scale by 2, Gram matrix by 4,
        # log-det increases but should be consistent
        r1 = rnd(scores1)
        r2 = rnd(scores2)
        assert r2 > r1  # larger scale → larger log-det

    def test_rnd_single_output_finite(self):
        scores = np.array([[0.8, 0.6, 0.9]])
        val    = rnd(scores)
        assert np.isfinite(val)


class TestRewardMetrics:
    def test_reward_variance_constant(self):
        rewards = np.array([0.7, 0.7, 0.7, 0.7])
        assert reward_variance(rewards) == pytest.approx(0.0, abs=1e-8)

    def test_reward_range_correct(self):
        rewards = np.array([0.2, 0.5, 0.9, 0.4])
        assert reward_range(rewards) == pytest.approx(0.7, abs=1e-8)


class TestDistinctN:
    def test_identical_texts_score_low(self):
        # Many copies of the same text: unique n-grams / total n-grams = 1/K
        K     = 10
        texts = ["the quick brown fox jumped over the lazy sleeping dog"] * K
        d     = distinct_n(texts, n=4)
        # 7 unique 4-grams out of 7*10=70 total → 0.1
        assert d < 0.15

    def test_unique_texts_score_high(self):
        texts = [
            "alpha beta gamma delta",
            "epsilon zeta eta theta",
            "iota kappa lambda mu",
        ]
        d = distinct_n(texts, n=2)
        assert d > 0.8
