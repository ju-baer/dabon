# Useful Diversity: A Predictive Geometry of Reward-Space Exploration for LLM Search
---

## One-Sentence Summary

We prove that naive geometric diversity is uncorrelated with Best-of-K gains,
derive a mechanistic account of when diversity *is* useful via reward-model
nullspaces, and build an adaptive sampler (**DABoN**) that outperforms uniform
exploration by learning to sample "where the reward model disagrees."

---

## Key Results

| Method | ΔBoK (Arena-Hard, N=16) | N₅₀ (budget for 50% oracle gain) |
|--------|------------------------|-----------------------------------|
| Greedy | 0.0 (baseline) | — |
| Standard BoN-16 | +5.4 | 8 |
| DPP-SBERT | +5.1 | 9 |
| **DABoN (ours)** | **+7.8** | **5** |

DABoN achieves **40–50% better sample efficiency** and higher gains across
GSM8K, MATH, HumanEval, and creative writing benchmarks.

---

## Repository Structure

```
dabon/
|
├── src/
│   ├── metrics/
│   │   └── diversity.py      # APCD, MMD, Self-BLEU, RND (Eq. 2)
│   ├── algorithms/
│   │   └── dabon.py          # DABoN selector + DPP kernel
│   ├── models/
│   │   ├── reward_ensemble.py # Ensemble wrapper (ArmoRM, Skywork, InternLM)
│   │   └── generation.py     # vLLM + HF generation wrapper
│   ├── data/
│   │   ├── dataset_loaders.py # GSM8K, MATH, Arena-Hard, HumanEval loaders
│   │   ├── gold_scorers.py   # Exact-match, sympy, unit-test, GPT-4o judge
│   │   └── pool_generator.py # Mixed-strategy candidate pool generation
│   ├── experiments/
│   │   ├── thrust1_correlation_audit.py    # Section 3: 16k sample audit
│   │   ├── thrust1_distractor_injection.py # Section 3.2: causal experiment
│   │   ├── thrust2_synthetic_manifold.py   # Section 4.4: blind-dim validation
│   │   ├── thrust3_dabon_evaluation.py     # Section 5: end-to-end benchmarks
│   │   └── thrust3_ablations.py            # Section 5.3: ablation study
│   └── utils/
│       └── embeddings.py     # SBERT + last-layer embedder
│
├── tests/
│   ├── test_diversity_metrics.py
│   ├── test_dabon.py
│   └── test_analysis.py
│
├── notebooks/
│   └── demo_and_figures.ipynb  # CPU-only demo, reproduces key figures
│
├── scripts/
│   ├── run_all_experiments.sh  # Master pipeline script
│   ├── generate_paper_figures.py
│   └── smoke_test.py           # Fast CPU sanity check
│
├── configs/                    # YAML configs for experiment sweeps
├── figures/                    # Generated PDF figures (gitignored)
├── results/                    # Experiment outputs (gitignored)
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/ju-baer/dabon.git
cd dabon
pip install -e ".[dev]"          # development install
pip install -e ".[all]"          # with vLLM, wandb, openai
```

### 2. Smoke test (CPU, ~30 seconds)

Validates all components before touching GPUs:

```bash
python scripts/smoke_test.py
```

### 3. Demo notebook (CPU, no models needed)

```bash
jupyter notebook notebooks/demo_and_figures.ipynb
```

Reproduces the synthetic blind-manifold results (Figure 3 equivalent)
and toy DABoN efficiency curves without any LLM inference.

### 4. Full pipeline (8×A100, ~1,800 GPU-hours)

```bash
# Smoke test first
python scripts/smoke_test.py

# Full run (all 4 tasks, 2000 prompts each)
bash scripts/run_all_experiments.sh

# Quick ablation run (arena_hard only, 100 prompts)
N_PROMPTS=100 bash scripts/run_all_experiments.sh --tasks arena_hard
```

### 5. Individual experiments

```bash
# Thrust I: correlation audit
python -m src.experiments.thrust1_correlation_audit \
    --tasks gsm8k arena_hard --model meta-llama/Llama-3.1-8B-Instruct \
    --n_prompts 500 --output_dir results/thrust1

# Thrust I: distractor injection (causal experiment)
python -m src.experiments.thrust1_distractor_injection \
    --task arena_hard --n_prompts 200

# Thrust II: synthetic manifold
python -m src.experiments.thrust2_synthetic_manifold \
    --n_sets 2000 --k 16

# Thrust III: end-to-end DABoN evaluation
python -m src.experiments.thrust3_dabon_evaluation \
    --task gsm8k --budgets 2 4 8 16 32 64

# Thrust III: ablation study
python -m src.experiments.thrust3_ablations \
    --task arena_hard --budget 16
```

---

## Using DABoN in Your Code

```python
import numpy as np
from src.algorithms.dabon import DABoNConfig, DABoNSelector

# Your reward ensemble scores: shape (M, m) where M=pool_size, m=ensemble_size
per_model_scores = np.random.rand(128, 3)   # replace with real RM scores
mean_rewards     = per_model_scores.mean(axis=1)
disagreement     = per_model_scores - mean_rewards[:, None]

cfg     = DABoNConfig(budget=16, adaptive_beta=True)
sel     = DABoNSelector(cfg)
indices = sel.select(mean_rewards, disagreement)

# indices: list of 16 selected indices into the pool
best_idx = indices[np.argmax(mean_rewards[np.array(indices)])]
print(f"Selected output index: {best_idx}")
```

---

## Computing the RND Metric

```python
from src.metrics.diversity import rnd

# reward_scores: (K, m) array of ensemble reward scores for K outputs
reward_scores = np.array([
    [0.9, 0.5, 0.7],
    [0.5, 0.9, 0.4],
    [0.3, 0.6, 0.9],
    ...
])
rnd_value = rnd(reward_scores)
print(f"RND = {rnd_value:.4f}")
```

---

## Reward Model Ensemble Setup

The paper uses ArmoRM, Skywork, and InternLM2 reward models:

```python
from src.models.reward_ensemble import build_ensemble

ensemble = build_ensemble(
    model_keys=["armo", "skywork", "internlm"],
    device="cuda",
)
results = ensemble.score_all(prompts, outputs)
# results["mean_scores"]:          (N,)
# results["disagreement_vectors"]: (N, 3)
```

For a cheaper approximation with MC-Dropout:

```python
from src.models.reward_ensemble import MCDropoutRewardModel, RewardModelConfig

cfg   = RewardModelConfig(model_name="RLHFlow/ArmoRM-Llama3-8B-v0.1")
model = MCDropoutRewardModel(cfg, n_samples=10, dropout_p=0.1)
mean_scores, score_samples = model.score_with_uncertainty(prompts, outputs)
# score_samples shape: (N, 10) — use as disagreement approximation
```


---

## Theoretical Contributions (Summary)

- **Theorem 1** (Useless Diversity): Geometric diversity in reward-flat
  directions has bounded BoK improvement.
- **Theorem 2** (RND as D-Optimal Design): Maximising RND is equivalent
  to D-optimal experimental design for learning the reward ensemble's
  epistemic errors.
- **Theorem 3** (BoK Lower Bound): RND lower-bounds expected BoK gain
  under sub-Gaussian epistemic noise.
- **Theorem 4** (Blind Dimension Sufficiency): Diversity in proxy-blind
  dimensions is necessary and sufficient for BoK gains when the true
  quality has components invisible to the proxy.
- **Theorem 5** (DABoN Guarantee): Greedy submodular selection achieves
  a $(1-1/e)$ approximation of the optimal diverse-quality set.

---

## License

MIT License. See `LICENSE` for details.
