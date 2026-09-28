"""
tests/test_analysis.py
-----------------------
Unit tests for statistical analysis utilities and the synthetic manifold.
"""

import numpy as np
import pytest
from src.experiments.thrust1_correlation_audit import (
    conditional_mutual_information,
    partial_correlation,
    unique_r2,
)
from src.experiments.thrust2_synthetic_manifold import (
    build_proxy_ensemble,
    proxy_quality,
    run_synthetic_experiment,
    sample_condition_A,
    sample_condition_B,
    sample_condition_C,
    true_quality,
)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

class TestPartialCorrelation:
    def test_positive_relationship(self):
        rng = np.random.default_rng(0)
        z   = rng.standard_normal(200)
        x   = z + rng.standard_normal(200) * 0.3
        y   = z + rng.standard_normal(200) * 0.3
        # After controlling for z, x and y should be weakly correlated
        rho = partial_correlation(x, y, z)
        assert abs(rho) < 0.3

    def test_zero_partial_correlation_when_independent(self):
        rng = np.random.default_rng(1)
        n   = 500
        z   = rng.standard_normal(n)
        x   = z + rng.standard_normal(n) * 0.1
        y   = rng.standard_normal(n)  # independent of x given z
        rho = partial_correlation(x, y, z)
        assert abs(rho) < 0.15

    def test_output_in_minus1_to_1(self):
        rng = np.random.default_rng(2)
        x, y, z = [rng.standard_normal(100) for _ in range(3)]
        rho = partial_correlation(x, y, z)
        assert -1.0 <= rho <= 1.0


class TestUniqueR2:
    def test_informative_feature_has_positive_unique_r2(self):
        rng = np.random.default_rng(3)
        n   = 300
        z   = rng.standard_normal(n)
        x   = rng.standard_normal(n)
        y   = 0.6 * x + 0.4 * z + rng.standard_normal(n) * 0.2
        u   = unique_r2(x, y, z)
        assert u > 0.05

    def test_uninformative_feature_near_zero(self):
        rng = np.random.default_rng(4)
        n   = 300
        z   = rng.standard_normal(n)
        x   = rng.standard_normal(n)   # independent of y
        y   = z + rng.standard_normal(n) * 0.1
        u   = unique_r2(x, y, z)
        assert u < 0.05


class TestCMI:
    def test_dependent_gt_independent(self):
        """
        CMI of dependent variables (x and y share a common cause z)
        should be lower after conditioning on z than CMI of x and y
        when x directly causes y.
        The key property: CMI(dependent) > CMI(independent).
        """
        rng = np.random.default_rng(5)
        n   = 800
        z   = rng.standard_normal(n)
        # Independent: x, y both depend only on z (no direct x->y link)
        x_ind = z + rng.standard_normal(n)
        y_ind = rng.standard_normal(n)        # independent of x given z
        # Dependent: x directly causes y (residual dependence given z)
        x_dep = z + rng.standard_normal(n) * 0.5
        y_dep = 2.0 * x_dep + rng.standard_normal(n) * 0.3
        cmi_ind = conditional_mutual_information(x_ind, y_ind, z)
        cmi_dep = conditional_mutual_information(x_dep, y_dep, z)
        # Dependent pair should have strictly higher CMI
        assert cmi_dep > cmi_ind, (
            f"CMI(dependent)={cmi_dep:.4f} should exceed CMI(independent)={cmi_ind:.4f}")

    def test_nonnegative(self):
        rng  = np.random.default_rng(6)
        x, y, z = [rng.standard_normal(100) for _ in range(3)]
        cmi  = conditional_mutual_information(x, y, z)
        assert cmi >= 0.0


# ---------------------------------------------------------------------------
# Synthetic manifold
# ---------------------------------------------------------------------------

class TestSyntheticManifold:
    def test_true_quality_shape(self):
        z  = np.random.rand(20, 4)
        r  = true_quality(z)
        assert r.shape == (20,)
        assert np.all(np.isfinite(r))

    def test_proxy_blind_to_hidden_dims(self):
        """Proxy should not differentiate on z₃, z₄."""
        rng = np.random.default_rng(7)
        z1  = rng.random((50, 4))
        z2  = z1.copy()
        z2[:, 2:] = rng.random((50, 2))  # shuffle hidden dims
        r1  = proxy_quality(z1, noise_std=0.0)
        r2  = proxy_quality(z2, noise_std=0.0)
        assert np.allclose(r1, r2, atol=1e-6), (
            "Proxy should be identical when only z₃, z₄ change"
        )

    def test_condition_B_low_hidden_spread(self):
        z = sample_condition_B(K=100)
        hidden_std = float(z[:, 2:].std())
        assert hidden_std < 0.1

    def test_condition_C_low_obs_spread(self):
        z = sample_condition_C(K=100)
        obs_std = float(z[:, :2].std())
        assert obs_std < 0.1

    def test_condition_A_full_spread(self):
        z = sample_condition_A(K=200)
        for dim in range(4):
            assert z[:, dim].std() > 0.3, f"Dim {dim} should have high spread"

    def test_synthetic_experiment_runs(self):
        """Smoke test: synthetic experiment completes and returns correct shape."""
        df = run_synthetic_experiment(n_sets=20, K=8, m=3, seed=99)
        assert len(df) == 20 * 3  # 3 conditions
        assert "bok_gain" in df.columns
        assert "rnd" in df.columns
        assert "condition" in df.columns

    def test_condition_C_efficiency(self):
        """
        Core claim: Condition C achieves BoK gain despite the proxy being
        blind to z3/z4 — measured as BoK-gain-per-proxy-variance ratio >> B.
        Validates Theorem 4 (Blind Dimension Sufficiency).
        """
        from src.experiments.thrust2_synthetic_manifold import (
            sample_condition_B, sample_condition_C, proxy_quality, bok_gain as bk
        )
        np.random.seed(0)
        K, N = 16, 150
        greedy = np.zeros(4)
        b_bok, c_bok, b_pvar, c_pvar = [], [], [], []
        for _ in range(N):
            zB = sample_condition_B(K)
            zC = sample_condition_C(K)
            b_bok.append(bk(zB, greedy))
            c_bok.append(bk(zC, greedy))
            b_pvar.append(float(np.std(proxy_quality(zB, 0.0))))
            c_pvar.append(float(np.std(proxy_quality(zC, 0.0))))
        # C gains despite proxy being blind (near-zero proxy variance)
        assert np.mean(c_bok) > 0.1, f"C BoK should be positive: {np.mean(c_bok):.4f}"
        assert np.mean(c_pvar) < np.mean(b_pvar) * 0.2, (
            f"C proxy std {np.mean(c_pvar):.4f} should be << B {np.mean(b_pvar):.4f}")
        ratio_B = np.mean(b_bok) / (np.mean(b_pvar) + 1e-9)
        ratio_C = np.mean(c_bok) / (np.mean(c_pvar) + 1e-9)
        assert ratio_C > ratio_B * 3, (
            f"C efficiency ratio {ratio_C:.2f} should be >> B {ratio_B:.2f}")

    def test_proxy_ensemble_shape(self):
        z = np.random.rand(10, 4)
        scores = build_proxy_ensemble(z, m=5)
        assert scores.shape == (10, 5)
