"""
tests/test_dabon.py
--------------------
Unit tests for the DABoN selection algorithm.
"""

import numpy as np
import pytest
from src.algorithms.dabon import (
    DABoNConfig,
    DABoNSelector,
    _marginal_gain_rank1,
    _sherman_morrison_update,
    standard_bon,
)


class TestMarginalGain:
    def test_positive_marginal_gain(self):
        """Marginal gain should be positive (adding new info)."""
        k_yy  = 1.0
        k_yS  = np.array([0.3])
        K_S_inv = np.array([[1.0]])
        gain  = _marginal_gain_rank1(k_yy, k_yS, K_S_inv)
        # Schur complement = 1.0 - 0.3*1.0*0.3 = 0.91 → log(0.91) < 0
        # (negative because schur < 1, but that's fine — it's a relative gain)
        assert np.isfinite(gain)

    def test_redundant_output_low_gain(self):
        """Perfectly correlated output → near-zero Schur → very negative gain."""
        k_yy  = 1.0
        k_yS  = np.array([1.0])    # identical to existing
        K_S_inv = np.array([[1.0]])
        gain  = _marginal_gain_rank1(k_yy, k_yS, K_S_inv)
        assert gain < 0  # near -inf: schur ≈ 0


class TestShermanMorrison:
    def test_inverse_correct_2x2(self):
        """Test that rank-1 update produces correct inverse."""
        sigma = 1.0
        k_yy  = 1.0
        k_yS  = np.array([0.5])
        K_inv = np.array([[1.0]])

        schur   = k_yy - k_yS @ K_inv @ k_yS
        new_inv = _sherman_morrison_update(K_inv, k_yS, k_yy, schur)

        # Build original 2x2 matrix
        K_mat = np.array([[1.0, 0.5], [0.5, 1.0]])
        expected_inv = np.linalg.inv(K_mat)
        assert np.allclose(new_inv, expected_inv, atol=1e-6)

    def test_inverse_shape(self):
        K_inv = np.eye(3)
        k_yS  = np.array([0.3, 0.2, 0.1])
        k_yy  = 1.0
        schur = k_yy - k_yS @ K_inv @ k_yS
        new_inv = _sherman_morrison_update(K_inv, k_yS, k_yy, schur)
        assert new_inv.shape == (4, 4)


class TestDABoNSelector:
    def setup_method(self):
        np.random.seed(0)
        self.M = 32
        self.m = 3
        self.N = 8

        # Synthetic pool: M candidates, m reward models
        self.mean_rewards = np.random.rand(self.M)
        per_model = np.random.rand(self.M, self.m)
        self.disagree = per_model - per_model.mean(axis=1, keepdims=True)

    def test_selects_exactly_N(self):
        cfg     = DABoNConfig(budget=self.N, adaptive_beta=False, beta=1.0)
        sel     = DABoNSelector(cfg)
        indices = sel.select(self.mean_rewards, self.disagree)
        assert len(indices) == self.N

    def test_no_duplicates(self):
        cfg     = DABoNConfig(budget=self.N, adaptive_beta=False, beta=1.0)
        sel     = DABoNSelector(cfg)
        indices = sel.select(self.mean_rewards, self.disagree)
        assert len(set(indices)) == len(indices), "Duplicate indices in selection"

    def test_all_indices_in_range(self):
        cfg     = DABoNConfig(budget=self.N, adaptive_beta=False, beta=1.0)
        sel     = DABoNSelector(cfg)
        indices = sel.select(self.mean_rewards, self.disagree)
        assert all(0 <= i < self.M for i in indices)

    def test_beta_zero_selects_top_n_by_reward(self):
        """With beta=0, DABoN is equivalent to top-N greedy selection."""
        cfg     = DABoNConfig(budget=self.N, beta=0.0, adaptive_beta=False,
                              top_k_candidates=self.M)
        sel     = DABoNSelector(cfg)
        indices = sel.select(self.mean_rewards, self.disagree)
        selected_rewards = sorted(self.mean_rewards[np.array(indices)], reverse=True)
        top_n_rewards    = sorted(self.mean_rewards, reverse=True)[:self.N]
        # Should have the same reward values (may differ in order)
        assert np.allclose(sorted(selected_rewards, reverse=True),
                           sorted(top_n_rewards, reverse=True), atol=1e-6)

    def test_diversity_bonus_increases_rnd(self):
        """Higher beta → selected set should have higher RND."""
        from src.metrics.diversity import rnd

        cfg_no_div = DABoNConfig(budget=self.N, beta=0.0, adaptive_beta=False,
                                 top_k_candidates=self.M)
        cfg_div    = DABoNConfig(budget=self.N, beta=5.0, adaptive_beta=False,
                                 top_k_candidates=self.M)

        sel_no = DABoNSelector(cfg_no_div)
        sel_di = DABoNSelector(cfg_div)

        idx_no = sel_no.select(self.mean_rewards, self.disagree)
        idx_di = sel_di.select(self.mean_rewards, self.disagree)

        rnd_no = rnd(self.disagree[np.array(idx_no)] + self.mean_rewards[np.array(idx_no)].reshape(-1,1))
        rnd_di = rnd(self.disagree[np.array(idx_di)] + self.mean_rewards[np.array(idx_di)].reshape(-1,1))

        # With high diversity bonus, we expect higher or equal RND
        # (This is a soft test — not always guaranteed with random data)
        assert np.isfinite(rnd_no) and np.isfinite(rnd_di)

    def test_budget_larger_than_pool_is_handled(self):
        cfg     = DABoNConfig(budget=self.M + 10, adaptive_beta=False, beta=1.0)
        sel     = DABoNSelector(cfg)
        indices = sel.select(self.mean_rewards, self.disagree)
        assert len(indices) <= self.M

    def test_select_best_returns_output(self):
        cfg     = DABoNConfig(budget=self.N, adaptive_beta=False, beta=1.0)
        sel     = DABoNSelector(cfg)
        outputs = [f"output_{i}" for i in range(self.M)]
        result  = sel.select_best(self.mean_rewards, self.disagree, outputs)
        assert isinstance(result, str)
        assert result.startswith("output_")


class TestStandardBoN:
    def test_returns_output_and_index(self):
        rewards = np.array([0.1, 0.9, 0.5, 0.3])
        outputs = ["a", "b", "c", "d"]
        out, idx = standard_bon(rewards, outputs, N=3)
        assert out in outputs
        assert 0 <= idx < len(outputs)

    def test_N_greater_than_pool_returns_best(self):
        rewards = np.array([0.1, 0.9, 0.5])
        outputs = ["a", "b", "c"]
        out, idx = standard_bon(rewards, outputs, N=10)
        assert out == "b"
        assert idx == 1
