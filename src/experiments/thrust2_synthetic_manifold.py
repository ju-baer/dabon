"""
src/experiments/thrust2_synthetic_manifold.py
----------------------------------------------
Thrust II: Synthetic validation of the Blind-Manifold Hypothesis.

Constructs a 4D latent space where:
  R*(z) = sin(2πz₁) + cos(2πz₂) + 0.5*ReLU(z₃) + ε
  R̂(z)  = sin(2πz₁) + cos(2πz₂)   [blind to z₃, z₄]

Tests three sampling conditions:
  A: diversity in all 4 dimensions  
  B: diversity only in observed (z₁, z₂)
  C: diversity only in hidden (z₃, z₄)

Expected: ΔBoK(A) ≈ ΔBoK(C) >> ΔBoK(B)

Also validates RND metric: compute RND using a proxy ensemble and check
its predictive correlation with BoK gain.

Usage:
    python -m src.experiments.thrust2_synthetic_manifold \
        --n_sets 1000 \
        --k 16 \
        --output_dir results/thrust2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.metrics.diversity import rnd, compute_disagreement_vectors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Ground truth functions
# ---------------------------------------------------------------------------

def true_quality(z: np.ndarray) -> np.ndarray:
    """
    R*(z) = sin(2πz₁) + cos(2πz₂) + 0.5 * ReLU(z₃) + ε
    Args:
        z: (N, 4) latent vectors
    Returns:
        (N,) quality scores
    """
    noise = np.random.randn(len(z)) * 0.05
    return (np.sin(2 * np.pi * z[:, 0])
            + np.cos(2 * np.pi * z[:, 1])
            + 0.5 * np.maximum(z[:, 2], 0.0)
            + noise)


def proxy_quality(z: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
    """
    R̂(z) = sin(2πz₁) + cos(2πz₂)  [blind to z₃, z₄]
    """
    noise = np.random.randn(len(z)) * noise_std
    return np.sin(2 * np.pi * z[:, 0]) + np.cos(2 * np.pi * z[:, 1]) + noise


def build_proxy_ensemble(
    z: np.ndarray,
    m: int = 3,
    noise_std: float = 0.1,
) -> np.ndarray:
    """
    Simulate m proxy reward models as noisy versions of proxy_quality.
    Returns (N, m) array.
    """
    scores = []
    for _ in range(m):
        scores.append(proxy_quality(z, noise_std=noise_std))
    return np.stack(scores, axis=1)


# ---------------------------------------------------------------------------
# Sampling conditions
# ---------------------------------------------------------------------------

def sample_condition_A(K: int) -> np.ndarray:
    """Full diversity in all 4 dimensions."""
    return np.random.uniform(-1, 1, (K, 4))


def sample_condition_B(K: int) -> np.ndarray:
    """Diversity only in observed dims (z₁, z₂); fixed z₃, z₄ ~ N(0, 0.01)."""
    z = np.zeros((K, 4))
    z[:, :2] = np.random.uniform(-1, 1, (K, 2))
    z[:, 2:] = np.random.randn(K, 2) * 0.01
    return z


def sample_condition_C(K: int) -> np.ndarray:
    """Diversity only in hidden dims (z₃, z₄); fixed z₁, z₂ ~ N(0, 0.01)."""
    z = np.zeros((K, 4))
    z[:, :2] = np.random.randn(K, 2) * 0.01
    z[:, 2:] = np.random.uniform(-1, 1, (K, 2))
    return z


CONDITION_SAMPLERS = {
    "A_full":    sample_condition_A,
    "B_observed": sample_condition_B,
    "C_hidden":  sample_condition_C,
}


# ---------------------------------------------------------------------------
# Compute BoK gain for a set
# ---------------------------------------------------------------------------

def bok_gain(z_set: np.ndarray, greedy_z: np.ndarray) -> float:
    """
    ΔBoK = max R*(z_set) - R*(greedy_z)
    """
    quality_set = true_quality(z_set)
    quality_greedy = true_quality(greedy_z.reshape(1, -1))[0]
    return float(quality_set.max() - quality_greedy)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_synthetic_experiment(
    n_sets: int = 1000,
    K: int = 16,
    m: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate n_sets sample sets for each condition and compute:
      - BoK gain
      - RND (using proxy ensemble)
      - geometric diversity (APCD on z coordinates)
    """
    np.random.seed(seed)
    rows = []

    # Fixed greedy point: origin (R*(0) = sin(0) + cos(0) = 1.0)
    greedy_z = np.zeros(4)

    for cond_name, sampler in CONDITION_SAMPLERS.items():
        logger.info("Running condition: %s", cond_name)
        for _ in range(n_sets):
            z_set = sampler(K)

            # Gold BoK gain
            gain = bok_gain(z_set, greedy_z)

            # Proxy ensemble scores
            reward_scores = build_proxy_ensemble(z_set, m=m, noise_std=0.1)

            # RND metric
            rnd_val = rnd(reward_scores)

            # Geometric APCD (on raw z coordinates, as proxy for SBERT)
            from src.metrics.diversity import average_pairwise_cosine_distance
            apcd_val = average_pairwise_cosine_distance(z_set)

            # Reward variance
            mean_r = reward_scores.mean(axis=1)
            rvar = float(np.std(mean_r))

            # Diversity in observed vs hidden dims
            obs_spread = float(np.std(z_set[:, :2]))
            hid_spread = float(np.std(z_set[:, 2:]))

            rows.append({
                "condition":     cond_name,
                "bok_gain":      gain,
                "rnd":           rnd_val,
                "apcd":          apcd_val,
                "reward_var":    rvar,
                "obs_spread":    obs_spread,
                "hid_spread":    hid_spread,
                "max_proxy":     float(mean_r.max()),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyse_results(df: pd.DataFrame) -> dict:
    """Summarise results by condition and compute key statistics."""
    summary = df.groupby("condition")["bok_gain"].agg(["mean", "std", "count"])
    logger.info("\nBoK Gain by condition:\n%s", summary.to_string())

    # Partial correlations per condition
    all_data = df.copy()
    from src.experiments.thrust1_correlation_audit import partial_correlation
    partial_rnd  = partial_correlation(
        all_data["rnd"].values,
        all_data["bok_gain"].values,
        all_data["max_proxy"].values,
    )
    partial_apcd = partial_correlation(
        all_data["apcd"].values,
        all_data["bok_gain"].values,
        all_data["max_proxy"].values,
    )
    partial_hid = partial_correlation(
        all_data["hid_spread"].values,
        all_data["bok_gain"].values,
        all_data["max_proxy"].values,
    )
    partial_obs = partial_correlation(
        all_data["obs_spread"].values,
        all_data["bok_gain"].values,
        all_data["max_proxy"].values,
    )

    logger.info(
        "Partial correlations (controlling for max_proxy):\n"
        "  RND:           %.3f\n"
        "  APCD:          %.3f\n"
        "  Hidden spread: %.3f\n"
        "  Observed spread: %.3f",
        partial_rnd, partial_apcd, partial_hid, partial_obs,
    )

    # Statistical test: condition A vs B vs C
    from scipy.stats import f_oneway
    bok_by_cond = {c: df[df["condition"] == c]["bok_gain"].values
                   for c in ["A_full", "B_observed", "C_hidden"]}
    f_stat, p_val = f_oneway(*bok_by_cond.values())
    logger.info("One-way ANOVA across conditions: F=%.2f, p=%.4f", f_stat, p_val)

    return {
        "condition_summary": summary,
        "partial_rnd":       partial_rnd,
        "partial_apcd":      partial_apcd,
        "partial_hid":       partial_hid,
        "partial_obs":       partial_obs,
        "anova_f":           f_stat,
        "anova_p":           p_val,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_condition_comparison(df: pd.DataFrame, output_dir: str):
    """Box plots of BoK gain by sampling condition."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    conds = ["A_full", "B_observed", "C_hidden"]
    labels = ["A: Full\n(all dims)", "B: Observed\n(z₁,z₂ only)", "C: Hidden\n(z₃,z₄ only)"]
    colors = ["#4878d0", "#ee854a", "#6acc65"]

    # Left: box plot
    ax = axes[0]
    data_by_cond = [df[df["condition"] == c]["bok_gain"].values for c in conds]
    bp = ax.boxplot(data_by_cond, patch_artist=True, labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("BoK Gain ($\\Delta_{\\mathrm{BoK}}$)", fontsize=11)
    ax.set_title("BoK Gain by Sampling Condition", fontsize=11)

    # Right: RND vs BoK scatter coloured by condition
    ax2 = axes[1]
    for cond, label, color in zip(conds, labels, colors):
        sub = df[df["condition"] == cond]
        ax2.scatter(sub["rnd"], sub["bok_gain"], alpha=0.3, s=8,
                    c=color, label=label.replace("\n", " "))
    ax2.set_xlabel("RND (Reward-Nullspace Diversity)", fontsize=11)
    ax2.set_ylabel("BoK Gain", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_title("RND Predicts BoK Gain", fontsize=11)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/synthetic_conditions.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Saved synthetic conditions figure.")


def plot_rnd_vs_geometric(df: pd.DataFrame, output_dir: str):
    """Side-by-side scatter: APCD vs BoK, RND vs BoK."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, xcol, xlabel in [
        (axes[0], "apcd", "Geometric Diversity (APCD)"),
        (axes[1], "rnd",  "RND (Reward-Nullspace Diversity)"),
    ]:
        ax.scatter(df[xcol], df["bok_gain"], alpha=0.15, s=6, c="steelblue")
        m, b, r, _, _ = stats.linregress(df[xcol], df["bok_gain"])
        xr = np.linspace(df[xcol].min(), df[xcol].max(), 100)
        ax.plot(xr, m * xr + b, "r--", lw=1.5, label=f"r={r:.2f}")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("BoK Gain", fontsize=11)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/rnd_vs_geometric_synthetic.pdf", bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = run_synthetic_experiment(
        n_sets=args.n_sets,
        K=args.k,
        m=args.m_ensemble,
        seed=args.seed,
    )
    df.to_csv(output_dir / "synthetic_results.csv", index=False)

    results = analyse_results(df)

    # Save summary stats
    results["condition_summary"].to_csv(output_dir / "condition_summary.csv")

    # Figures
    fig_dir = str(output_dir / "figures")
    plot_condition_comparison(df, fig_dir)
    plot_rnd_vs_geometric(df, fig_dir)

    logger.info("Thrust II complete. Results in %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sets",      type=int, default=1000)
    parser.add_argument("--k",           type=int, default=16)
    parser.add_argument("--m_ensemble",  type=int, default=3)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_dir",  default="results/thrust2")
    args = parser.parse_args()
    main(args)
