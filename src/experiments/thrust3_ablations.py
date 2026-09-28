"""
src/experiments/thrust3_ablations.py
--------------------------------------
DABoN ablation study (Table 3 in paper).

Variants tested:
  - DABoN full (adaptive beta, ensemble disagreement)
  - beta=0 (no diversity bonus → pure greedy-by-reward)
  - Random Δ vectors (not from ensemble)
  - m=1 single reward model (no disagreement)
  - Constant beta (not adaptive)
  - Small pool M=N (no selection advantage)
  - Geometric kernel (SBERT cosine, not disagreement)

Usage:
    python -m src.experiments.thrust3_ablations \
        --task arena_hard \
        --model llama3-8b \
        --n_prompts 200 \
        --budget 16 \
        --output_dir results/thrust3/ablations
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.algorithms.dabon import DABoNConfig, DABoNSelector
from src.data.dataset_loaders import load_task_prompts
from src.data.gold_scorers import get_gold_scorer
from src.data.pool_generator import generate_pool
from src.models.reward_ensemble import build_ensemble
from src.metrics.diversity import compute_disagreement_vectors, rnd
from src.utils.embeddings import SBERTEmbedder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation_variant(
    variant_name: str,
    mean_rewards: np.ndarray,
    disagreement_vectors: np.ndarray,
    embeddings: np.ndarray,
    gold_scores: np.ndarray,
    greedy_gold: float,
    budget: int,
) -> float:
    """
    Evaluate one ablation variant on a single prompt's pool.
    Returns absolute BoK gain.
    """

    if variant_name == "DABoN_full":
        cfg = DABoNConfig(budget=budget, beta=1.0, adaptive_beta=True)
        sel = DABoNSelector(cfg)
        indices = sel.select(mean_rewards, disagreement_vectors)

    elif variant_name == "beta_zero":
        # Pure greedy by mean reward — no diversity bonus
        cfg = DABoNConfig(budget=budget, beta=0.0, adaptive_beta=False)
        sel = DABoNSelector(cfg)
        indices = sel.select(mean_rewards, disagreement_vectors)

    elif variant_name == "random_delta":
        # Random disagreement vectors (not from ensemble)
        np.random.seed(42)
        rand_deltas = np.random.randn(*disagreement_vectors.shape)
        cfg = DABoNConfig(budget=budget, beta=1.0, adaptive_beta=False)
        sel = DABoNSelector(cfg)
        indices = sel.select(mean_rewards, rand_deltas)

    elif variant_name == "single_rm":
        # Use only first RM score → disagreement is all zeros
        single_scores = mean_rewards.reshape(-1, 1)
        zero_deltas   = np.zeros((len(mean_rewards), 1))
        # Fall back to beta=0 behaviour
        cfg = DABoNConfig(budget=budget, beta=0.0, adaptive_beta=False)
        sel = DABoNSelector(cfg)
        indices = sel.select(mean_rewards, zero_deltas)

    elif variant_name == "const_beta":
        cfg = DABoNConfig(budget=budget, beta=1.0, adaptive_beta=False)
        sel = DABoNSelector(cfg)
        indices = sel.select(mean_rewards, disagreement_vectors)

    elif variant_name == "small_pool":
        # Simulate M=N by restricting to first `budget` candidates
        cfg = DABoNConfig(budget=budget, beta=1.0, adaptive_beta=True)
        sel = DABoNSelector(cfg)
        perm = np.random.permutation(len(mean_rewards))[:budget]
        local_indices = sel.select(
            mean_rewards[perm], disagreement_vectors[perm]
        )
        indices = [perm[i] for i in local_indices]

    elif variant_name == "geometric_kernel":
        # Replace disagreement vectors with SBERT embeddings (geometric kernel)
        # Normalise embeddings to unit sphere for comparability
        norms   = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        normed  = embeddings / norms
        cfg = DABoNConfig(budget=budget, beta=1.0, adaptive_beta=True)
        sel = DABoNSelector(cfg)
        indices = sel.select(mean_rewards, normed)

    elif variant_name == "standard_bon":
        perm    = np.random.permutation(len(mean_rewards))[:budget]
        best    = perm[np.argmax(mean_rewards[perm])]
        indices = [best]

    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    if not indices:
        return 0.0

    best_local = int(np.argmax(mean_rewards[np.array(indices)]))
    best_idx   = indices[best_local]
    return float(gold_scores[best_idx] - greedy_gold)


ABLATION_VARIANTS = [
    "DABoN_full",
    "beta_zero",
    "random_delta",
    "single_rm",
    "const_beta",
    "small_pool",
    "geometric_kernel",
    "standard_bon",
]

VARIANT_LABELS = {
    "DABoN_full":       "DABoN (full)",
    "beta_zero":        r"$\beta=0$ (no diversity)",
    "random_delta":     "Random $\\Delta$ (not ensemble)",
    "single_rm":        "$m=1$ (single RM)",
    "const_beta":       "Constant $\\beta$",
    "small_pool":       "Small pool ($M=N$)",
    "geometric_kernel": "Geometric kernel (SBERT)",
    "standard_bon":     "Standard BoN",
}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts  = load_task_prompts(args.task, n=args.n_prompts)
    ensemble = build_ensemble(["armo", "skywork", "internlm"], device=args.device)
    embedder = SBERTEmbedder()
    gold_fn  = get_gold_scorer(args.task)

    results: Dict[str, List[float]] = defaultdict(list)

    for prompt in tqdm(prompts, desc="Ablation prompts"):
        try:
            pool = generate_pool(prompt, args.model, pool_size=args.pool_size,
                                 device=args.device)
            if not pool:
                continue

            rm_out      = ensemble.score_all([prompt] * len(pool), pool)
            mean_r      = rm_out["mean_scores"]
            disagree    = rm_out["disagreement_vectors"]
            embeddings  = embedder.embed(pool)
            gold_scores = np.array([gold_fn(prompt, o) for o in pool])
            greedy_gold = gold_scores[0]

            for variant in ABLATION_VARIANTS:
                try:
                    gain = run_ablation_variant(
                        variant_name=variant,
                        mean_rewards=mean_r,
                        disagreement_vectors=disagree,
                        embeddings=embeddings,
                        gold_scores=gold_scores,
                        greedy_gold=greedy_gold,
                        budget=args.budget,
                    )
                    results[variant].append(gain)
                except Exception as e:
                    logger.debug("Variant %s failed: %s", variant, e)

        except Exception as e:
            logger.warning("Skipping prompt: %s", e)

    # Build summary table
    rows = []
    dabon_gains = np.array(results["DABoN_full"])
    dabon_mean  = dabon_gains.mean() if len(dabon_gains) else 1.0

    for variant in ABLATION_VARIANTS:
        gains = np.array(results[variant])
        if len(gains) == 0:
            continue
        mean_g = gains.mean()
        std_g  = gains.std()
        delta  = mean_g - dabon_mean
        rows.append({
            "Variant":      VARIANT_LABELS[variant],
            "Mean ΔBoK":    f"{mean_g:.3f}",
            "Std":          f"{std_g:.3f}",
            "Δ vs DABoN":   f"{delta:+.3f}",
        })

    df_table = pd.DataFrame(rows)
    print("\nAblation Study Results:")
    print(df_table.to_string(index=False))
    df_table.to_csv(output_dir / "ablation_table.csv", index=False)

    # Bar chart
    _plot_ablation_bars(results, output_dir)
    logger.info("Ablations complete. Results in %s", output_dir)


def _plot_ablation_bars(
    results: Dict[str, List[float]],
    output_dir: Path,
):
    import matplotlib.pyplot as plt

    variants  = ABLATION_VARIANTS
    means     = [np.mean(results[v]) if results[v] else 0.0 for v in variants]
    sems      = [
        np.std(results[v]) / max(len(results[v]), 1)**0.5
        if results[v] else 0.0 for v in variants
    ]
    labels    = [VARIANT_LABELS[v] for v in variants]
    colors    = [
        "#4878d0" if v == "DABoN_full" else "#aec7e8" for v in variants
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(variants))
    bars = ax.bar(x, means, yerr=[1.96 * s for s in sems],
                  capsize=4, color=colors, edgecolor="black",
                  linewidth=0.7, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel(r"Mean $\Delta_{\mathrm{BoK}}$", fontsize=11)
    ax.set_title("DABoN Ablation Study", fontsize=12)
    ax.axhline(means[0], color="#4878d0", ls="--", lw=1.0, alpha=0.5,
               label="DABoN full")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path = output_dir / "figures" / "ablation_bars.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fig_path), bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved ablation figure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       default="arena_hard")
    parser.add_argument("--model",      default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_prompts",  type=int, default=200)
    parser.add_argument("--budget",     type=int, default=16)
    parser.add_argument("--pool_size",  type=int, default=128)
    parser.add_argument("--output_dir", default="results/thrust3/ablations")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    main(args)
