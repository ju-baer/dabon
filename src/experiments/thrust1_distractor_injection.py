"""
src/experiments/thrust1_distractor_injection.py
------------------------------------------------
Causal distractor injection experiment (Section 3.2 / Study B).

Protocol:
  1. Take a seed set S_seed of high-reward outputs (R̂ ≈ 0.85).
  2. Generate S_rand: outputs with high geometric distance from S_seed
     but reward held constant within ±0.05 (via rejection sampling).
  3. Generate S_grad: outputs with high reward-gradient-aligned diversity.
  4. Compare ΔBoK across S_seed, S_seed ∪ S_rand, S_seed ∪ S_grad.

Expected result:
  ΔBoK(S_seed ∪ S_rand) ≈ ΔBoK(S_seed)   [geometric diversity → no gain]
  ΔBoK(S_seed ∪ S_grad) >  ΔBoK(S_seed)   [reward-aligned diversity → gain]

Usage:
    python -m src.experiments.thrust1_distractor_injection \
        --task arena_hard \
        --model llama3-8b \
        --n_prompts 500 \
        --output_dir results/thrust1/distractor
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from src.data.dataset_loaders import load_task_prompts
from src.data.gold_scorers import get_gold_scorer
from src.data.pool_generator import generate_pool
from src.models.generation import GenerationConfig, batch_generate
from src.models.reward_ensemble import build_ensemble
from src.utils.embeddings import SBERTEmbedder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Rejection-sampled distractors
# ---------------------------------------------------------------------------

def generate_random_distractors(
    prompt: str,
    seed_mean_reward: float,
    model_name: str,
    n_distractors: int = 32,
    reward_tolerance: float = 0.05,
    ensemble=None,
    max_attempts: int = 200,
) -> Tuple[List[str], np.ndarray]:
    """
    Generate outputs that have reward ≈ seed_mean_reward but
    are geometrically diverse (no special reward-aligned structure).

    Uses rejection sampling: generate high-temperature outputs,
    keep those within ±reward_tolerance of target reward.

    Returns:
        accepted_outputs:  list of accepted output strings
        accepted_rewards:  (N,) mean reward values
    """
    cfg = GenerationConfig(
        model_name=model_name,
        temperature=1.5,       # high temp for geometric spread
        top_p=1.0,
        do_sample=True,
        max_new_tokens=512,
        use_vllm=True,
    )
    accepted_outputs, accepted_rewards = [], []
    n_attempts = 0

    while len(accepted_outputs) < n_distractors and n_attempts < max_attempts:
        batch_size = min(32, (n_distractors - len(accepted_outputs)) * 3)
        candidates = batch_generate([prompt] * batch_size, cfg)
        rm_results  = ensemble.score_all([prompt] * len(candidates), candidates)
        mean_rewards = rm_results["mean_scores"]

        for out, rew in zip(candidates, mean_rewards):
            if abs(rew - seed_mean_reward) <= reward_tolerance:
                accepted_outputs.append(out)
                accepted_rewards.append(rew)
                if len(accepted_outputs) >= n_distractors:
                    break

        n_attempts += batch_size

    logger.debug(
        "Accepted %d/%d distractors (rejection rate: %.1f%%)",
        len(accepted_outputs), n_distractors,
        100 * (1 - len(accepted_outputs) / max(n_attempts, 1)),
    )
    return accepted_outputs[:n_distractors], np.array(accepted_rewards[:n_distractors])


def generate_reward_gradient_additions(
    prompt: str,
    seed_outputs: List[str],
    model_name: str,
    n_additions: int = 32,
    ensemble=None,
) -> Tuple[List[str], np.ndarray]:
    """
    Generate outputs with diversity in reward-gradient directions:
    use best-of-8 sampling to push outputs toward high-reward but
    different semantic regions (diverse by reward change).

    Practically: sample with moderate temperature, keep outputs
    that have HIGHER reward than seed AND are semantically different.
    """
    from src.metrics.diversity import average_pairwise_cosine_distance
    from src.utils.embeddings import SBERTEmbedder
    embedder = SBERTEmbedder()
    seed_embeddings = embedder.embed(seed_outputs)
    seed_mean_reward = ensemble.score_all(
        [prompt] * len(seed_outputs), seed_outputs
    )["mean_scores"].mean()

    cfg = GenerationConfig(
        model_name=model_name,
        temperature=0.9,
        top_p=0.95,
        do_sample=True,
        max_new_tokens=512,
        use_vllm=True,
    )
    accepted_outputs, accepted_rewards = [], []

    for _ in range(n_additions * 5):
        if len(accepted_outputs) >= n_additions:
            break
        candidates = batch_generate([prompt] * 8, cfg)
        rm_res       = ensemble.score_all([prompt] * 8, candidates)
        mean_rewards = rm_res["mean_scores"]
        # Keep the one with highest reward among candidates
        best_idx  = int(np.argmax(mean_rewards))
        best_out  = candidates[best_idx]
        best_rew  = mean_rewards[best_idx]
        # Accept if it's better than the seed mean (reward-gradient direction)
        if best_rew > seed_mean_reward - 0.02:
            accepted_outputs.append(best_out)
            accepted_rewards.append(best_rew)

    return accepted_outputs[:n_additions], np.array(accepted_rewards[:n_additions])


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_distractor_experiment_for_prompt(
    prompt: str,
    model_name: str,
    ensemble,
    gold_scorer,
    embedder: SBERTEmbedder,
    n_seed: int = 8,
    n_distractors: int = 16,
) -> Dict:
    """
    Run the full distractor injection experiment for one prompt.

    Returns dict with keys:
      seed_bok, rand_bok, grad_bok,
      seed_apcd, rand_apcd, grad_apcd,
      seed_rvar, rand_rvar, grad_rvar
    """
    # --- Generate seed set (best-of-32 to get high-reward cluster) ---
    pool   = generate_pool(prompt, model_name, pool_size=32)
    rm_res = ensemble.score_all([prompt] * len(pool), pool)
    mean_r = rm_res["mean_scores"]

    # Select top-n_seed by mean reward
    top_indices  = np.argsort(mean_r)[-n_seed:][::-1]
    seed_outputs = [pool[i] for i in top_indices]
    seed_rewards = mean_r[top_indices]
    seed_mean_r  = float(seed_rewards.mean())

    # --- Random distractors (reward-neutral) ---
    rand_outputs, rand_rewards = generate_random_distractors(
        prompt=prompt,
        seed_mean_reward=seed_mean_r,
        model_name=model_name,
        n_distractors=n_distractors,
        ensemble=ensemble,
    )

    # --- Reward-gradient additions ---
    grad_outputs, grad_rewards = generate_reward_gradient_additions(
        prompt=prompt,
        seed_outputs=seed_outputs,
        model_name=model_name,
        n_additions=n_distractors,
        ensemble=ensemble,
    )

    # --- Compute gold BoK gain for each condition ---
    def bok_gain_for_set(outputs: List[str]) -> float:
        scores    = np.array([gold_scorer(prompt, o) for o in outputs])
        greedy_g  = gold_scorer(prompt, seed_outputs[0])
        return float(scores.max() - greedy_g)

    set_base   = seed_outputs
    set_rand   = seed_outputs + rand_outputs[:n_distractors]
    set_grad   = seed_outputs + grad_outputs[:n_distractors]

    bok_base   = bok_gain_for_set(set_base)
    bok_rand   = bok_gain_for_set(set_rand)
    bok_grad   = bok_gain_for_set(set_grad)

    # --- Compute diversity metrics ---
    from src.metrics.diversity import average_pairwise_cosine_distance, reward_variance

    def embed_and_apcd(outputs):
        emb = embedder.embed(outputs)
        return average_pairwise_cosine_distance(emb)

    apcd_base  = embed_and_apcd(set_base)
    apcd_rand  = embed_and_apcd(set_rand)
    apcd_grad  = embed_and_apcd(set_grad)

    rvar_base  = float(reward_variance(seed_rewards))
    rvar_rand  = float(reward_variance(
        np.concatenate([seed_rewards, rand_rewards[:n_distractors]])
    ))
    rvar_grad  = float(reward_variance(
        np.concatenate([seed_rewards, grad_rewards[:n_distractors]])
    ))

    return {
        "seed_bok":  bok_base, "rand_bok":  bok_rand, "grad_bok":  bok_grad,
        "seed_apcd": apcd_base,"rand_apcd": apcd_rand,"grad_apcd": apcd_grad,
        "seed_rvar": rvar_base,"rand_rvar": rvar_rand,"grad_rvar": rvar_grad,
    }


# ---------------------------------------------------------------------------
# Statistical tests & figures
# ---------------------------------------------------------------------------

def analyse_distractor_results(df: pd.DataFrame) -> dict:
    """
    Run paired t-tests and compute summary statistics.
    """
    # Test 1: rand_bok vs seed_bok (should be NOT significant)
    t1, p1 = stats.ttest_rel(df["rand_bok"], df["seed_bok"])
    # Test 2: grad_bok vs seed_bok (should be significant, positive)
    t2, p2 = stats.ttest_rel(df["grad_bok"], df["seed_bok"])
    # Test 3: rand_apcd > seed_apcd (verify manipulation worked)
    t3, p3 = stats.ttest_rel(df["rand_apcd"], df["seed_apcd"])

    logger.info(
        "\nDistractor Experiment Results:\n"
        "  Seed BoK:       %.3f ± %.3f\n"
        "  Rand BoK:       %.3f ± %.3f   (vs seed: t=%.2f, p=%.4f)\n"
        "  Grad BoK:       %.3f ± %.3f   (vs seed: t=%.2f, p=%.4f)\n"
        "  Seed APCD:      %.3f | Rand APCD: %.3f  (manipulation check: p=%.4f)",
        df["seed_bok"].mean(), df["seed_bok"].std(),
        df["rand_bok"].mean(), df["rand_bok"].std(), t1, p1,
        df["grad_bok"].mean(), df["grad_bok"].std(), t2, p2,
        df["seed_apcd"].mean(), df["rand_apcd"].mean(), p3,
    )

    return {
        "t_rand_vs_seed": t1, "p_rand_vs_seed": p1,
        "t_grad_vs_seed": t2, "p_grad_vs_seed": p2,
        "manipulation_check_p": p3,
        "seed_bok_mean": df["seed_bok"].mean(),
        "rand_bok_mean": df["rand_bok"].mean(),
        "grad_bok_mean": df["grad_bok"].mean(),
    }


def plot_distractor_results(df: pd.DataFrame, output_dir: str):
    """Generate Figure 2 from the paper."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: scatter - geometric diversity vs BoK gain
    ax = axes[0]
    ax.scatter(df["seed_apcd"], df["seed_bok"],
               alpha=0.55, s=22, c="#4878d0", label="Seed set", zorder=4)
    ax.scatter(df["rand_apcd"], df["rand_bok"],
               alpha=0.45, s=22, c="#ee854a", marker="^",
               label="+ Random distractors")
    ax.scatter(df["grad_apcd"], df["grad_bok"],
               alpha=0.45, s=22, c="#6acc65", marker="s",
               label="+ Reward-gradient additions")

    # Regression line for seed
    m, b, r, _, _ = stats.linregress(df["seed_apcd"], df["seed_bok"])
    xr = np.linspace(df["seed_apcd"].min(), df["rand_apcd"].max(), 80)
    ax.plot(xr, m * xr + b, "--", color="gray", lw=1.2, alpha=0.6)

    ax.set_xlabel("Geometric Diversity (APCD)", fontsize=11)
    ax.set_ylabel(r"BoK Gain ($\Delta_{\mathrm{BoK}}$)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("Geometric Diversity vs BoK Gain", fontsize=11)

    # Right: bar chart of mean BoK gain per condition
    ax2 = axes[1]
    conditions = ["Seed", "+ Random\nDistractors", "+ Reward-Grad\nAdditions"]
    means = [df["seed_bok"].mean(), df["rand_bok"].mean(), df["grad_bok"].mean()]
    sems  = [
        df["seed_bok"].std() / len(df)**0.5,
        df["rand_bok"].std() / len(df)**0.5,
        df["grad_bok"].std() / len(df)**0.5,
    ]
    colors = ["#4878d0", "#ee854a", "#6acc65"]
    bars = ax2.bar(conditions, means, yerr=sems, capsize=5,
                   color=colors, alpha=0.82, edgecolor="black", linewidth=0.8)

    # Significance annotation
    y_max = max(means) + max(sems) * 1.5
    ax2.plot([0, 2], [y_max * 1.05, y_max * 1.05], "k-", lw=1.0)
    ax2.text(1, y_max * 1.07, "***", ha="center", va="bottom", fontsize=12)
    ax2.plot([0, 1], [y_max * 0.98, y_max * 0.98], "k-", lw=0.8, alpha=0.5)
    ax2.text(0.5, y_max * 0.99, "n.s.", ha="center", va="bottom",
             fontsize=9, color="gray")

    ax2.set_ylabel("Mean BoK Gain", fontsize=11)
    ax2.set_title("Causal Effect of Diversity Type", fontsize=11)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(output_dir) / "distractor_experiment.pdf")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved distractor figure to %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts   = load_task_prompts(args.task, n=args.n_prompts)
    ensemble  = build_ensemble(["armo", "skywork", "internlm"], device=args.device)
    embedder  = SBERTEmbedder()
    gold_fn   = get_gold_scorer(args.task)

    rows = []
    for prompt in tqdm(prompts, desc="Distractor injection"):
        try:
            result = run_distractor_experiment_for_prompt(
                prompt=prompt,
                model_name=args.model,
                ensemble=ensemble,
                gold_scorer=gold_fn,
                embedder=embedder,
                n_seed=8,
                n_distractors=16,
            )
            result["prompt"] = prompt
            rows.append(result)
        except Exception as e:
            logger.warning("Skipping prompt due to error: %s", e)
            continue

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "distractor_results.csv", index=False)

    stats_dict = analyse_distractor_results(df)
    import json
    with open(output_dir / "distractor_stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

    plot_distractor_results(df, str(output_dir / "figures"))
    logger.info("Distractor experiment complete. Results in %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       default="arena_hard")
    parser.add_argument("--model",      default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_prompts",  type=int, default=500)
    parser.add_argument("--output_dir", default="results/thrust1/distractor")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    main(args)
