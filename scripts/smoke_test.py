"""
scripts/smoke_test.py
----------------------
Fast smoke test (~30 seconds, CPU only) that validates all core
components are importable and functionally correct before running
the full pipeline on GPUs.

Usage:
    python scripts/smoke_test.py
"""

import sys
import traceback
import numpy as np

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
errors = []

def check(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        errors.append((name, traceback.format_exc()))

print("\n=== DABoN Smoke Test ===\n")

# 1. Imports
print("[ Imports ]")
check("metrics.diversity", lambda: __import__("src.metrics.diversity"))
check("algorithms.dabon",  lambda: __import__("src.algorithms.dabon"))
check("data.dataset_loaders", lambda: __import__("src.data.dataset_loaders"))
check("data.gold_scorers", lambda: __import__("src.data.gold_scorers"))
check("utils.embeddings",  lambda: __import__("src.utils.embeddings"))

# 2. Diversity metrics
print("\n[ Diversity Metrics ]")
from src.metrics.diversity import (
    average_pairwise_cosine_distance, rnd,
    compute_disagreement_vectors, distinct_n, reward_variance
)

def test_apcd():
    emb = np.eye(5, dtype=float)
    d = average_pairwise_cosine_distance(emb)
    assert abs(d - 1.0) < 1e-5, f"Expected 1.0, got {d}"

def test_rnd_monotone():
    low  = np.tile([0.9, 0.5, 0.7], (8, 1))
    high = np.random.default_rng(0).random((8, 3))
    r_l = rnd(low); r_h = rnd(high)
    assert r_h > r_l, f"Diverse should have higher RND: {r_h:.3f} vs {r_l:.3f}"

def test_disagreement_zero_mean():
    scores = np.random.rand(10, 3)
    d = compute_disagreement_vectors(scores)
    assert np.allclose(d.sum(axis=1), 0, atol=1e-9)

check("APCD orthogonal vectors = 1.0", test_apcd)
check("RND monotone in diversity",     test_rnd_monotone)
check("Disagreement vectors zero-sum", test_disagreement_zero_mean)

# 3. DABoN algorithm
print("\n[ DABoN Algorithm ]")
from src.algorithms.dabon import DABoNConfig, DABoNSelector, standard_bon, _sherman_morrison_update

def test_dabon_selects_N():
    rng = np.random.default_rng(1)
    M, m, N = 50, 3, 10
    mr = rng.random(M)
    dv = rng.random((M, m)) - 0.5
    cfg = DABoNConfig(budget=N, beta=1.0, adaptive_beta=False, top_k_candidates=M)
    sel = DABoNSelector(cfg)
    idx = sel.select(mr, dv)
    assert len(idx) == N, f"Expected {N}, got {len(idx)}"
    assert len(set(idx)) == N, "Duplicate indices"

def test_dabon_budget_exceeds_pool():
    rng = np.random.default_rng(2)
    mr = rng.random(5)
    dv = rng.random((5, 3)) - 0.5
    cfg = DABoNConfig(budget=20, adaptive_beta=False)
    sel = DABoNSelector(cfg)
    idx = sel.select(mr, dv)
    assert len(idx) <= 5

def test_sherman_morrison_2x2():
    K_inv = np.array([[1.0]])
    k_yS  = np.array([0.5])
    k_yy  = 1.0
    schur = k_yy - k_yS @ K_inv @ k_yS
    new_inv = _sherman_morrison_update(K_inv, k_yS, k_yy, schur)
    expected = np.linalg.inv(np.array([[1.0, 0.5], [0.5, 1.0]]))
    assert np.allclose(new_inv, expected, atol=1e-6)

check("DABoN selects exactly N",          test_dabon_selects_N)
check("DABoN handles budget > pool",      test_dabon_budget_exceeds_pool)
check("Sherman-Morrison 2×2 inverse",     test_sherman_morrison_2x2)

# 4. Synthetic manifold
print("\n[ Synthetic Manifold ]")
from src.experiments.thrust2_synthetic_manifold import (
    true_quality, proxy_quality, sample_condition_B, sample_condition_C,
    run_synthetic_experiment
)

def test_proxy_blind():
    rng = np.random.default_rng(3)
    z1  = rng.random((20, 4))
    z2  = z1.copy(); z2[:, 2:] = rng.random((20, 2))
    assert np.allclose(proxy_quality(z1, 0.0), proxy_quality(z2, 0.0), atol=1e-6)

def test_synth_experiment_shape():
    df = run_synthetic_experiment(n_sets=10, K=8, m=3, seed=0)
    assert len(df) == 30, f"Expected 30 rows (3 conds × 10), got {len(df)}"

def test_condition_C_efficiency():
    """
    Real claim: Condition C achieves positive BoK gain despite near-zero
    proxy variance — BoK-per-proxy-var efficiency ratio >> condition B.
    This validates Theorem 4 (Blind Dimension Sufficiency).
    """
    from src.experiments.thrust2_synthetic_manifold import (
        sample_condition_B, sample_condition_C, proxy_quality,
        bok_gain as bk
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
    assert np.mean(c_bok) > 0.1, f"C BoK should be positive, got {np.mean(c_bok):.4f}"
    assert np.mean(c_pvar) < np.mean(b_pvar) * 0.2, (
        f"C proxy std {np.mean(c_pvar):.4f} should be << B {np.mean(b_pvar):.4f}")
    ratio_B = np.mean(b_bok) / (np.mean(b_pvar) + 1e-9)
    ratio_C = np.mean(c_bok) / (np.mean(c_pvar) + 1e-9)
    assert ratio_C > ratio_B * 3, (
        f"C efficiency ratio {ratio_C:.2f} should be >> B {ratio_B:.2f}")

check("Proxy blind to hidden dims",                test_proxy_blind)
check("Synthetic experiment shape",                test_synth_experiment_shape)
check("Condition C: BoK/proxy-var efficiency >> B", test_condition_C_efficiency)

# 5. Statistical analysis
print("\n[ Statistical Analysis ]")
from src.experiments.thrust1_correlation_audit import (
    partial_correlation, unique_r2, conditional_mutual_information
)

def test_partial_corr_range():
    rng  = np.random.default_rng(4)
    x, y, z = [rng.standard_normal(200) for _ in range(3)]
    rho  = partial_correlation(x, y, z)
    assert -1.0 <= rho <= 1.0

def test_unique_r2_nonneg():
    rng  = np.random.default_rng(5)
    x, y, z = [rng.standard_normal(100) for _ in range(3)]
    u    = unique_r2(x, y, z)
    assert u >= 0.0

check("Partial correlation in [-1,1]",   test_partial_corr_range)
check("Unique R² non-negative",           test_unique_r2_nonneg)

# 6. Data loaders
print("\n[ Data & Scoring ]")
from src.data.dataset_loaders import load_task_prompts
from src.data.gold_scorers import get_gold_scorer

def test_loader_returns_list():
    prompts = load_task_prompts("gsm8k", n=5)
    assert isinstance(prompts, list)
    assert len(prompts) == 5

def test_scorer_callable():
    fn = get_gold_scorer("gsm8k")
    score = fn("what is 2+2?", "The answer is 4.")
    assert isinstance(score, float)

check("Dataset loader returns list",     test_loader_returns_list)
check("Gold scorer is callable",         test_scorer_callable)

# =============================================================================
print("\n" + "="*40)
if errors:
    print(f"\033[91m{len(errors)} test(s) FAILED:\033[0m")
    for name, tb in errors:
        print(f"\n  [{name}]\n{tb}")
    sys.exit(1)
else:
    print(f"\033[92mAll smoke tests passed!\033[0m")
    print("Ready to run: bash scripts/run_all_experiments.sh --dry-run")
