"""
src/experiments/thrust1_correlation_audit.py
---------------------------------------------
Thrust I: The Negative Result
  - Generate 16,000 sample sets across 4 tasks × 8 samplers × 2,000 prompts
  - Compute all diversity metrics
  - Run partial correlation analysis and mutual information estimation
  - Produce Figure 1 (scatter plots) and Table 1 (correlation table)

Usage:
    python -m src.experiments.thrust1_correlation_audit \
        --tasks gsm8k arena_hard humaneval creative_writing \
        --model llama3-8b \
        --output_dir results/thrust1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm

from src.metrics.diversity import compute_all_metrics
from src.data.dataset_loaders import load_task_prompts
from src.data.gold_scorers import get_gold_scorer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Sampling strategies (Section 3.1.2)
# ---------------------------------------------------------------------------

SAMPLING_STRATEGIES = {
    "temp_0.3":  {"temperature": 0.3, "do_sample": True, "top_p": 1.0},
    "temp_0.6":  {"temperature": 0.6, "do_sample": True, "top_p": 1.0},
    "temp_1.0":  {"temperature": 1.0, "do_sample": True, "top_p": 1.0},
    "temp_1.5":  {"temperature": 1.5, "do_sample": True, "top_p": 1.0},
    "nucleus_0.85": {"temperature": 1.0, "do_sample": True, "top_p": 0.85},
    "nucleus_0.95": {"temperature": 1.0, "do_sample": True, "top_p": 0.95},
    "dbs_0.5":   {"num_beams": 8, "num_beam_groups": 8, "diversity_penalty": 0.5},
    "dbs_2.0":   {"num_beams": 8, "num_beam_groups": 8, "diversity_penalty": 2.0},
}


# ---------------------------------------------------------------------------
# Partial correlation (Equation 1 in paper)
# ---------------------------------------------------------------------------

def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Compute partial correlation ρ(x, y | z) after linearly controlling for z.
    """
    def residual(a, b):
        # regress a on b, return residuals
        b_ = b.reshape(-1, 1)
        coef = np.linalg.lstsq(
            np.column_stack([b_, np.ones(len(b_))]), a, rcond=None
        )[0]
        return a - (b_ * coef[0] + coef[1]).flatten()

    r_xz = residual(x, z)
    r_yz = residual(y, z)
    if r_xz.std() < 1e-10 or r_yz.std() < 1e-10:
        return 0.0
    rho, _ = stats.pearsonr(r_xz, r_yz)
    return float(rho)


def unique_r2(
    diversity: np.ndarray,
    bok_gain: np.ndarray,
    max_reward: np.ndarray,
) -> float:
    """
    Unique variance in BoK gain explained by diversity metric,
    beyond what max_reward already explains.
    
    ΔR² = R²(diversity + max_reward) - R²(max_reward only)
    """
    X_full = np.column_stack([diversity, max_reward])
    X_base = max_reward.reshape(-1, 1)
    reg_full = LinearRegression().fit(X_full, bok_gain)
    reg_base = LinearRegression().fit(X_base, bok_gain)
    r2_full = reg_full.score(X_full, bok_gain)
    r2_base = reg_base.score(X_base, bok_gain)
    return max(0.0, r2_full - r2_base)


# ---------------------------------------------------------------------------
# Conditional mutual information via histogram binning
# ---------------------------------------------------------------------------

def conditional_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_neighbors: int = 5,
) -> float:
    """
    I(x; y | z) estimated via the CMI decomposition:
      CMI(x; y | z) = MI(x; y, z) - MI(x; z)
    using sklearn's k-NN mutual information estimator, which is far
    more reliable than histogram binning on continuous data.
    """
    from sklearn.feature_selection import mutual_info_regression
    # Stack z and (y, z) as feature matrices
    yz = np.column_stack([y, z])
    z_ = z.reshape(-1, 1)
    mi_x_yz = mutual_info_regression(yz, x, n_neighbors=n_neighbors, random_state=0)[0]
    # MI(x; z) using just z
    mi_x_z  = mutual_info_regression(z_, x, n_neighbors=n_neighbors, random_state=0)[0]
    return float(max(0.0, mi_x_yz - mi_x_z))


# ---------------------------------------------------------------------------
# Generate sample sets for one task
# ---------------------------------------------------------------------------

def generate_sample_sets(
    prompts: List[str],
    model_name: str,
    strategies: Dict,
    samples_per_strategy: int = 16,
    output_dir: Optional[str] = None,
) -> List[Dict]:
    """
    For each (prompt, strategy) pair generate `samples_per_strategy` outputs.
    Returns list of dicts: {prompt, strategy, outputs, gold_score}.
    """
    results = []
    for strategy_name, sample_kwargs in tqdm(
        strategies.items(), desc="Strategies"
    ):
        logger.info("Strategy: %s", strategy_name)
        gen_cfg = GenerationConfig(
            model_name=model_name,
            max_new_tokens=512,
            **sample_kwargs,
        )
        for prompt in tqdm(prompts, desc=f"  Prompts [{strategy_name}]", leave=False):
            outputs = batch_generate(
                [prompt] * samples_per_strategy,
                gen_cfg,
            )
            results.append({
                "prompt": prompt,
                "strategy": strategy_name,
                "outputs": outputs,
            })
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_dir, "raw_outputs.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    return results


# ---------------------------------------------------------------------------
# Compute metrics for all sample sets
# ---------------------------------------------------------------------------

def build_feature_matrix(
    sample_sets: List[Dict],
    ensemble,
    embed_fn,
    gold_score_fn,
    reference_embeddings: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    For each sample set, compute all diversity metrics and gold BoK gain.
    Returns a DataFrame with one row per sample set.
    """
    rows = []
    for item in tqdm(sample_sets, desc="Computing metrics"):
        prompt   = item["prompt"]
        outputs  = item["outputs"]
        strategy = item["strategy"]

        # Embeddings
        embeddings = embed_fn(outputs)                   # (K, d)

        # Reward scores
        prompts_rep = [prompt] * len(outputs)
        rm_result   = ensemble.score_all(prompts_rep, outputs)
        reward_scores = rm_result["per_model_scores"]    # (K, m)
        mean_rewards  = rm_result["mean_scores"]         # (K,)

        # Gold BoK gain
        gold_scores = np.array([gold_score_fn(prompt, o) for o in outputs])
        greedy_gold = gold_score_fn(prompt, outputs[0])  # first = greedy approx
        bok_gain    = float(gold_scores.max() - greedy_gold)

        # Diversity metrics
        metrics = compute_all_metrics(
            texts=outputs,
            embeddings=embeddings,
            reward_scores=reward_scores,
            reference_embeddings=reference_embeddings,
        )
        metrics["bok_gain"] = bok_gain
        metrics["strategy"] = strategy
        metrics["prompt"]   = prompt
        rows.append(metrics)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def run_correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute partial correlations, unique R², and CMI for all diversity metrics.
    """
    diversity_cols = ["apcd", "mmd", "self_bleu", "distinct_4",
                      "reward_var", "reward_range", "rnd"]
    diversity_cols = [c for c in diversity_cols if c in df.columns]

    bok   = df["bok_gain"].values
    maxr  = df["max_reward"].values
    rows  = []

    for col in diversity_cols:
        div = df[col].values
        raw_r, _  = stats.pearsonr(div, bok)
        partial_r = partial_correlation(div, bok, maxr)
        u_r2      = unique_r2(div, bok, maxr)
        cmi       = conditional_mutual_information(div, bok, maxr)
        rows.append({
            "metric":      col,
            "raw_rho":     raw_r,
            "partial_rho": partial_r,
            "unique_r2":   u_r2,
            "cmi_nats":    cmi,
        })

    return pd.DataFrame(rows).sort_values("partial_rho", ascending=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_scatter_comparison(
    df: pd.DataFrame,
    output_dir: str,
):
    """Figure 1: scatter plots of geometric vs RND diversity against BoK gain."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    pairs = [
        ("apcd",  "Geometric Diversity (APCD)", "tab:red"),
        ("reward_var", "Reward Variance",        "tab:orange"),
        ("rnd",   "RND (ours)",                  "tab:blue"),
    ]

    for ax, (col, label, color) in zip(axes, pairs):
        if col not in df.columns:
            continue
        x = df[col].values
        y = df["bok_gain"].values
        ax.scatter(x, y, alpha=0.25, s=8, color=color)
        # Fit line
        m, b, r, _, _ = stats.linregress(x, y)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, m * xr + b, color="black", lw=1.5, ls="--",
                label=f"r={r:.2f}")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("BoK Gain ($\\Delta_{\\mathrm{BoK}}$)", fontsize=11)
        ax.legend(fontsize=10)
        ax.set_title(label, fontsize=11)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "scatter_diversity_vs_bok.pdf"),
                bbox_inches="tight")
    plt.close()
    logger.info("Saved scatter figure.")


def plot_distractor_experiment(
    base_bok: np.ndarray,
    rand_bok: np.ndarray,
    grad_bok: np.ndarray,
    base_geo: np.ndarray,
    rand_geo: np.ndarray,
    grad_geo: np.ndarray,
    output_dir: str,
):
    """Figure 2: distractor injection experiment."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: geometric diversity vs BoK gain
    ax = axes[0]
    ax.scatter(base_geo, base_bok, alpha=0.5, s=20, c="tab:blue",
               label="Seed set", zorder=3)
    ax.scatter(rand_geo, rand_bok, alpha=0.5, s=20, c="tab:red",
               marker="^", label="+ Random distractors")
    ax.scatter(grad_geo, grad_bok, alpha=0.5, s=20, c="tab:green",
               marker="s", label="+ Reward-gradient additions")
    ax.set_xlabel("Geometric Diversity (APCD)", fontsize=11)
    ax.set_ylabel("BoK Gain", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title("Distractor Injection", fontsize=11)

    # Right: bar chart of mean BoK gain
    ax2 = axes[1]
    means = [base_bok.mean(), rand_bok.mean(), grad_bok.mean()]
    sems  = [base_bok.std() / len(base_bok)**0.5,
             rand_bok.std() / len(rand_bok)**0.5,
             grad_bok.std() / len(grad_bok)**0.5]
    bars = ax2.bar(["Seed", "+Random", "+Reward-Grad"], means,
                   yerr=sems, capsize=4,
                   color=["tab:blue", "tab:red", "tab:green"], alpha=0.8)
    ax2.set_ylabel("Mean BoK Gain", fontsize=11)
    ax2.set_title("Mean Gain by Condition", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distractor_experiment.pdf"),
                bbox_inches="tight")
    plt.close()
    logger.info("Saved distractor figure.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    all_sample_sets = []
    for task in args.tasks:
        logger.info("Loading task: %s", task)
        prompts = load_task_prompts(task, n=args.n_prompts)
        sets = generate_sample_sets(
            prompts=prompts,
            model_name=args.model,
            strategies=SAMPLING_STRATEGIES,
            samples_per_strategy=16,
            output_dir=str(output_dir / "raw" / task),
        )
        all_sample_sets.extend(sets)

    # Load ensemble
    ensemble = build_ensemble(
        ["armo", "skywork", "internlm"],
        device=args.device,
    )

    # Embed texts
    from src.utils.embeddings import SBERTEmbedder
    embedder = SBERTEmbedder()

    # Gold scoring (task-dependent stub; see src/data/gold_scorers.py)
    from src.data.gold_scorers import get_gold_scorer
    gold_fn = get_gold_scorer(args.tasks[0])

    # Build feature matrix
    df = build_feature_matrix(
        sample_sets=all_sample_sets,
        ensemble=ensemble,
        embed_fn=embedder.embed,
        gold_score_fn=gold_fn,
    )
    df.to_csv(output_dir / "features.csv", index=False)
    logger.info("Saved features.csv with %d rows", len(df))

    # Statistical analysis
    stats_df = run_correlation_analysis(df)
    stats_df.to_csv(output_dir / "correlation_results.csv", index=False)
    print(stats_df.to_string(float_format="%.3f"))

    # Figures
    plot_scatter_comparison(df, str(output_dir / "figures"))

    logger.info("Thrust I complete. Results in %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+",
                        default=["gsm8k", "arena_hard"])
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_prompts", type=int, default=500)
    parser.add_argument("--output_dir", default="results/thrust1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)
