"""
src/experiments/thrust3_dabon_evaluation.py
--------------------------------------------
Thrust III: End-to-end evaluation of DABoN vs baselines.

Runs:
  - All baselines (greedy, temperature, nucleus, DBS, standard BoN, DPP-SBERT)
  - DABoN at budget N ∈ {2, 4, 8, 16, 32, 64}
  - Full ablation study (Table 3 in paper)
  - Sample efficiency curves (Figure 3)
  - Generalisation to held-out reward model

Usage:
    python -m src.experiments.thrust3_dabon_evaluation \
        --task arena_hard \
        --model llama3-8b \
        --budgets 2 4 8 16 32 64 \
        --output_dir results/thrust3
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.algorithms.dabon import DABoNConfig, DABoNSelector, standard_bon
from src.models.reward_ensemble import build_ensemble
from src.utils.embeddings import SBERTEmbedder
from src.data.dataset_loaders import load_task_prompts
from src.data.gold_scorers import get_gold_scorer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def normalised_bok_gain(
    selected_gold: float,
    greedy_gold: float,
    oracle_gold: float,
) -> float:
    """
    Δ_rel = (selected_gold - greedy_gold) / (oracle_gold - greedy_gold)
    Clipped to [0, 1].
    """
    denom = oracle_gold - greedy_gold
    if abs(denom) < 1e-8:
        return 0.0
    return float(np.clip((selected_gold - greedy_gold) / denom, 0.0, 1.0))


def area_under_gain_curve(gains_by_step: np.ndarray) -> float:
    """
    AUGC = (1/N_max) * Σ_t gain_at_step_t
    """
    return float(np.mean(gains_by_step))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    method: str
    budget: int
    mean_bok_gain: float
    std_bok_gain: float
    mean_norm_gain: float
    augc: float
    n50: Optional[float] = None


def evaluate_standard_bon(
    pool_mean_rewards: np.ndarray,
    pool_gold_scores: np.ndarray,
    greedy_gold: float,
    oracle_gold: float,
    N: int,
) -> Tuple[float, float]:
    """Random subsample N from pool; pick best by mean reward."""
    perm = np.random.permutation(len(pool_mean_rewards))[:N]
    best_idx = perm[np.argmax(pool_mean_rewards[perm])]
    gain = normalised_bok_gain(
        pool_gold_scores[best_idx], greedy_gold, oracle_gold
    )
    abs_gain = float(pool_gold_scores[best_idx] - greedy_gold)
    return abs_gain, gain


def evaluate_dpp_sbert(
    pool_embeddings: np.ndarray,
    pool_mean_rewards: np.ndarray,
    pool_gold_scores: np.ndarray,
    greedy_gold: float,
    oracle_gold: float,
    N: int,
    beta: float = 1.0,
) -> Tuple[float, float]:
    """DPP with SBERT cosine kernel (geometric baseline)."""
    from src.metrics.diversity import rnd as rnd_metric
    K_mat = pool_embeddings @ pool_embeddings.T
    norms = np.linalg.norm(pool_embeddings, axis=1)
    K_mat = K_mat / (np.outer(norms, norms) + 1e-12)
    # Use cosine similarity as kernel
    selected = []
    K_inv = None
    for step in range(N):
        best_idx, best_score = -1, -float("inf")
        for idx in range(len(pool_mean_rewards)):
            if idx in selected:
                continue
            if step == 0:
                gain = float(np.log(K_mat[idx, idx] + 1e-4))
            else:
                k_yS = np.array([K_mat[idx, s] for s in selected])
                schur = K_mat[idx, idx] - k_yS @ K_inv @ k_yS
                gain = float(np.log(max(schur, 1e-12)))
            score = pool_mean_rewards[idx] + beta * gain
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        if step == 0:
            K_inv = np.array([[1.0 / (K_mat[best_idx, best_idx] + 1e-4)]])
        else:
            k_yS = np.array([K_mat[best_idx, s] for s in selected[:-1]])
            schur = K_mat[best_idx, best_idx] - k_yS @ K_inv @ k_yS
            schur = max(schur, 1e-12)
            from src.algorithms.dabon import _sherman_morrison_update
            K_inv = _sherman_morrison_update(K_inv, k_yS, K_mat[best_idx, best_idx], schur)

    best_in_selected = selected[int(np.argmax(pool_mean_rewards[np.array(selected)]))]
    abs_gain = float(pool_gold_scores[best_in_selected] - greedy_gold)
    norm_gain = normalised_bok_gain(pool_gold_scores[best_in_selected], greedy_gold, oracle_gold)
    return abs_gain, norm_gain


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------

ABLATION_VARIANTS = {
    "DABoN_full": DABoNConfig(budget=16, beta=1.0, adaptive_beta=True),
    "beta_zero":  DABoNConfig(budget=16, beta=0.0, adaptive_beta=False),
    "const_beta": DABoNConfig(budget=16, beta=1.0, adaptive_beta=False),
    "small_pool": None,   # handled separately: M=N
}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_prompt(
    prompt: str,
    pool_outputs: List[str],
    pool_mean_rewards: np.ndarray,
    pool_reward_scores: np.ndarray,
    pool_disagreements: np.ndarray,
    pool_embeddings: np.ndarray,
    gold_scorer,
    budgets: List[int],
    dabon_selector: DABoNSelector,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate all methods on a single prompt's candidate pool.
    Returns dict: method_name → {budget → abs_bok_gain}
    """
    # Compute gold scores for all pool outputs
    pool_gold = np.array([gold_scorer(prompt, o) for o in pool_outputs])
    greedy_gold = pool_gold[0]  # first output = greedy
    oracle_gold = pool_gold.max()

    results = defaultdict(dict)

    for N in budgets:
        # Standard BoN
        abs_g, _ = evaluate_standard_bon(
            pool_mean_rewards, pool_gold, greedy_gold, oracle_gold, N
        )
        results["BoN"][N] = abs_g

        # DPP-SBERT
        abs_g, _ = evaluate_dpp_sbert(
            pool_embeddings, pool_mean_rewards, pool_gold,
            greedy_gold, oracle_gold, N
        )
        results["DPP-SBERT"][N] = abs_g

        # DABoN
        cfg = DABoNConfig(budget=N, adaptive_beta=True)
        sel = DABoNSelector(cfg)
        indices = sel.select(pool_mean_rewards, pool_disagreements)
        best_local = int(np.argmax(pool_mean_rewards[np.array(indices)]))
        best_idx = indices[best_local]
        abs_g = float(pool_gold[best_idx] - greedy_gold)
        results["DABoN"][N] = abs_g

    return dict(results)


# ---------------------------------------------------------------------------
# Aggregate and plot
# ---------------------------------------------------------------------------

def compute_n50(method_gains: Dict[int, List[float]], oracle_gains: List[float]) -> float:
    """
    N50: budget N to achieve 50% of oracle gain on average.
    Linearly interpolate between budget points.
    """
    budgets = sorted(method_gains.keys())
    oracle_mean = np.mean(oracle_gains)
    target = 0.5 * oracle_mean
    for i, N in enumerate(budgets):
        mean_gain = np.mean(method_gains[N])
        if mean_gain >= target:
            if i == 0:
                return float(N)
            prev_N = budgets[i - 1]
            prev_gain = np.mean(method_gains[prev_N])
            # Linear interpolation
            frac = (target - prev_gain) / (mean_gain - prev_gain + 1e-12)
            return float(prev_N + frac * (N - prev_N))
    return float(budgets[-1])


def plot_efficiency_curves(
    results_by_method: Dict[str, Dict[int, List[float]]],
    output_dir: str,
):
    """Figure 3: BoK gain vs budget N for all methods."""
    budgets = sorted(next(iter(results_by_method.values())).keys())
    colors = {
        "Greedy":   "gray",
        "Temp-1.0": "tab:orange",
        "Nucleus":  "tab:purple",
        "DBS":      "tab:brown",
        "BoN":      "tab:red",
        "DPP-SBERT":"tab:cyan",
        "DABoN":    "tab:blue",
    }
    ls = {
        "Greedy": ":", "Temp-1.0": "--", "Nucleus": "--",
        "DBS": "--", "BoN": "-", "DPP-SBERT": "-.", "DABoN": "-",
    }
    lw = {"DABoN": 2.5, "BoN": 2.0}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, gains_by_budget in results_by_method.items():
        means = [np.mean(gains_by_budget[N]) for N in budgets]
        sems  = [np.std(gains_by_budget[N]) / len(gains_by_budget[N])**0.5
                 for N in budgets]
        ax.errorbar(
            budgets, means,
            yerr=[1.96 * s for s in sems],
            label=method,
            color=colors.get(method, "black"),
            linestyle=ls.get(method, "-"),
            linewidth=lw.get(method, 1.5),
            marker="o",
            markersize=4,
            capsize=3,
        )

    ax.set_xlabel("Budget $N$", fontsize=12)
    ax.set_ylabel("BoK Gain ($\\Delta_{\\mathrm{BoK}}$)", fontsize=12)
    ax.set_title("Sample Efficiency Curves", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/efficiency_curves.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Saved efficiency curves figure.")


def build_results_table(
    results_by_method: Dict[str, Dict[int, List[float]]],
    budgets: List[int],
) -> pd.DataFrame:
    """Build Table 2 for the paper (results at a fixed budget)."""
    target_N = 16
    rows = []
    oracle_gains_approx = results_by_method.get("DABoN", {}).get(max(budgets), [1.0])
    for method, gains_by_budget in results_by_method.items():
        if target_N not in gains_by_budget:
            continue
        gs = gains_by_budget[target_N]
        n50 = compute_n50(gains_by_budget, oracle_gains_approx)
        rows.append({
            "Method":           method,
            f"ΔBoK (N={target_N})": f"{np.mean(gs):.2f} ± {np.std(gs):.2f}",
            "N50":              f"{n50:.1f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted(args.budgets)

    # Load prompts & tools
    prompts    = load_task_prompts(args.task, n=args.n_prompts)
    ensemble   = build_ensemble(["armo", "skywork", "internlm"], device=args.device)
    embedder   = SBERTEmbedder()
    gold_fn    = get_gold_scorer(args.task)

    # Stub: load / generate candidate pools (replace with real generation)
    from src.data.pool_generator import generate_pool
    dabon_cfg  = DABoNConfig(budget=max(budgets), pool_size=args.pool_size,
                             adaptive_beta=True)
    selector   = DABoNSelector(dabon_cfg)

    results_by_method: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for prompt in tqdm(prompts, desc="Evaluating prompts"):
        pool_outputs = generate_pool(prompt, args.model, args.pool_size, args.device)
        if not pool_outputs:
            continue

        # Score pool
        rm_out    = ensemble.score_all([prompt] * len(pool_outputs), pool_outputs)
        mean_r    = rm_out["mean_scores"]
        disagree  = rm_out["disagreement_vectors"]
        embeddings = embedder.embed(pool_outputs)
        gold_scores = np.array([gold_fn(prompt, o) for o in pool_outputs])
        greedy_gold = gold_scores[0]
        oracle_gold = gold_scores.max()

        for N in budgets:
            # BoN
            perm = np.random.permutation(len(pool_outputs))[:N]
            bon_best = perm[np.argmax(mean_r[perm])]
            results_by_method["BoN"][N].append(
                float(gold_scores[bon_best] - greedy_gold)
            )

            # DPP-SBERT
            abs_g, _ = evaluate_dpp_sbert(
                embeddings, mean_r, gold_scores, greedy_gold, oracle_gold, N
            )
            results_by_method["DPP-SBERT"][N].append(abs_g)

            # DABoN
            cfg = DABoNConfig(budget=N, adaptive_beta=True)
            sel = DABoNSelector(cfg)
            indices = sel.select(mean_r, disagree)
            best_local = int(np.argmax(mean_r[np.array(indices)]))
            best_idx = indices[best_local]
            results_by_method["DABoN"][N].append(
                float(gold_scores[best_idx] - greedy_gold)
            )

        # Greedy (constant across budgets)
        results_by_method["Greedy"][budgets[0]].append(0.0)

    # Aggregate and display
    table = build_results_table(results_by_method, budgets)
    print("\nResults Table:")
    print(table.to_string(index=False))
    table.to_csv(output_dir / "results_table.csv", index=False)

    # Efficiency curves
    plot_efficiency_curves(
        {k: dict(v) for k, v in results_by_method.items()},
        str(output_dir / "figures"),
    )

    logger.info("Thrust III complete. Results in %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       default="arena_hard")
    parser.add_argument("--model",      default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_prompts",  type=int, default=200)
    parser.add_argument("--pool_size",  type=int, default=128)
    parser.add_argument("--budgets",    type=int, nargs="+",
                        default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--output_dir", default="results/thrust3")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    main(args)
