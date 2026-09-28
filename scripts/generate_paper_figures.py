"""
scripts/generate_paper_figures.py
-----------------------------------
Aggregates all experiment results and generates the final camera-ready figures
for the paper. Run after all three experiment thrusts complete.

Usage:
    python scripts/generate_paper_figures.py \
        --results_dir results/ \
        --output_dir figures/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "pdf.fonttype": 42,   # ensures fonts are embedded
    "ps.fonttype": 42,
})

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Figure 1: Scatter — geometric vs RND diversity vs BoK gain (Table 1 vis.)
# ---------------------------------------------------------------------------

def fig1_scatter_comparison(results_dir: Path, output_dir: Path):
    """Three-panel scatter: APCD, D_var, RND vs BoK gain."""
    csv_path = results_dir / "thrust1" / "audit" / "features.csv"
    if not csv_path.exists():
        logger.warning("Figure 1 data missing: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    panels = [
        ("apcd",       "Geometric Diversity (APCD)",     "#ee854a"),
        ("reward_var", "Reward Variance",                 "#f19143"),
        ("rnd",        "RND (ours)",                      "#4878d0"),
    ]
    for ax, (col, label, color) in zip(axes, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        x, y = df[col].dropna().values, df.loc[df[col].notna(), "bok_gain"].values
        ax.scatter(x, y, alpha=0.2, s=6, c=color, rasterized=True)
        m, b, r, _, _ = stats.linregress(x, y)
        xr = np.linspace(x.min(), x.max(), 200)
        ax.plot(xr, m * xr + b, "k--", lw=1.5, label=f"$r={r:.2f}$")
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\Delta_{\mathrm{BoK}}$")
        ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_scatter_diversity.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 1.")


# ---------------------------------------------------------------------------
# Figure 2: Distractor injection (causal experiment)
# ---------------------------------------------------------------------------

def fig2_distractor(results_dir: Path, output_dir: Path):
    csv_path = results_dir / "thrust1" / "distractor" / "distractor_results.csv"
    if not csv_path.exists():
        logger.warning("Figure 2 data missing: %s", csv_path)
        return

    df   = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left scatter
    ax = axes[0]
    ax.scatter(df["seed_apcd"], df["seed_bok"],  alpha=0.5, s=16,
               c="#4878d0", label="Seed set", zorder=4)
    ax.scatter(df["rand_apcd"], df["rand_bok"],  alpha=0.4, s=16,
               c="#ee854a", marker="^", label="+ Random distractors")
    ax.scatter(df["grad_apcd"], df["grad_bok"],  alpha=0.4, s=16,
               c="#6acc65", marker="s", label="+ Reward-grad additions")
    ax.set_xlabel("Geometric Diversity (APCD)")
    ax.set_ylabel(r"BoK Gain ($\Delta_{\mathrm{BoK}}$)")
    ax.legend()
    ax.set_title("Geometric Diversity vs BoK Gain")

    # Right bar chart
    ax2 = axes[1]
    labels = ["Seed", "+Random\nDistractors", "+Reward-Grad\nAdditions"]
    means  = [df["seed_bok"].mean(), df["rand_bok"].mean(), df["grad_bok"].mean()]
    sems   = [df[c].std() / len(df)**0.5
              for c in ["seed_bok", "rand_bok", "grad_bok"]]
    colors = ["#4878d0", "#ee854a", "#6acc65"]
    ax2.bar(labels, means, yerr=[1.96*s for s in sems], capsize=5,
            color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    y_ann = max(means) + max(sems) * 2.5
    ax2.annotate("***", xy=(2, means[2] + sems[2] * 2),
                 ha="center", fontsize=14)
    ax2.annotate("n.s.", xy=(1, means[1] + sems[1] * 2),
                 ha="center", fontsize=10, color="gray")
    ax2.set_ylabel("Mean BoK Gain")
    ax2.set_title("Causal Effect of Diversity Type")

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_distractor.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 2.")


# ---------------------------------------------------------------------------
# Figure 3: Sample efficiency curves
# ---------------------------------------------------------------------------

def fig3_efficiency(results_dir: Path, output_dir: Path):
    """Load all tasks' efficiency results and produce multi-panel figure."""
    tasks = ["gsm8k", "arena_hard", "humaneval", "creative_writing"]
    task_labels = {"gsm8k": "GSM8K", "arena_hard": "Arena-Hard",
                   "humaneval": "HumanEval", "creative_writing": "Creative Writing"}

    methods = {
        "BoN":      ("#ee854a", "--"),
        "DPP-SBERT":("#a9c5e8", "-."),
        "DABoN":    ("#4878d0", "-"),
    }

    n_tasks = sum(
        1 for t in tasks
        if (results_dir / "thrust3" / t / "results_table.csv").exists()
    )
    if n_tasks == 0:
        logger.warning("No Thrust III data found.")
        return

    fig, axes = plt.subplots(1, max(n_tasks, 1), figsize=(4.5 * n_tasks, 4.2),
                             sharey=False)
    if n_tasks == 1:
        axes = [axes]

    ax_idx = 0
    for task in tasks:
        csv = results_dir / "thrust3" / task / "results_table.csv"
        eff_csv = results_dir / "thrust3" / task / "efficiency.csv"
        if not csv.exists():
            continue
        ax = axes[ax_idx]
        ax_idx += 1

        # Try to load detailed efficiency data
        if eff_csv.exists():
            eff = pd.read_csv(eff_csv)
            for method, (color, ls) in methods.items():
                sub = eff[eff["method"] == method]
                if sub.empty:
                    continue
                budgets = sorted(sub["budget"].unique())
                means   = [sub[sub["budget"] == N]["mean_gain"].values[0]
                           for N in budgets]
                sems    = [sub[sub["budget"] == N]["sem_gain"].values[0]
                           for N in budgets]
                ax.errorbar(budgets, means, yerr=[1.96*s for s in sems],
                            label=method, color=color, ls=ls, lw=2,
                            marker="o", markersize=4, capsize=2)

        ax.set_xlabel("Budget $N$")
        ax.set_ylabel(r"$\Delta_{\mathrm{BoK}}$")
        ax.set_title(task_labels.get(task, task))
        ax.set_xscale("log", base=2)
        if ax_idx == 1:
            ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_efficiency.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 3.")


# ---------------------------------------------------------------------------
# Figure 4: Ablation bars
# ---------------------------------------------------------------------------

def fig4_ablations(results_dir: Path, output_dir: Path):
    csv_path = results_dir / "thrust3" / "ablations" / "ablation_table.csv"
    if not csv_path.exists():
        logger.warning("Figure 4 data missing: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 4.2))

    # Parse mean ± std strings
    means, stds = [], []
    for val in df["Mean ΔBoK"]:
        parts = str(val).split("±")
        means.append(float(parts[0].strip()))
        stds.append(float(parts[1].strip()) if len(parts) > 1 else 0.0)

    x      = np.arange(len(df))
    colors = ["#4878d0" if "DABoN (full)" in str(v) else "#aec7e8"
              for v in df["Variant"]]
    ax.bar(x, means, yerr=[1.96 * s for s in stds], capsize=4,
           color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)
    ax.axhline(means[0], color="#4878d0", ls="--", lw=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Variant"].tolist(), rotation=20, ha="right")
    ax.set_ylabel(r"Mean $\Delta_{\mathrm{BoK}}$")
    ax.set_title("DABoN Ablation Study (Arena-Hard, $N=16$)")

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_ablations.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 4.")


# ---------------------------------------------------------------------------
# Supplementary: Synthetic manifold figures
# ---------------------------------------------------------------------------

def fig_supp_synthetic(results_dir: Path, output_dir: Path):
    csv_path = results_dir / "thrust2" / "synthetic" / "synthetic_results.csv"
    if not csv_path.exists():
        logger.warning("Synthetic data missing: %s", csv_path)
        return

    from src.experiments.thrust2_synthetic_manifold import (
        plot_condition_comparison, plot_rnd_vs_geometric
    )
    df = pd.read_csv(csv_path)
    plot_condition_comparison(df, str(output_dir))
    plot_rnd_vs_geometric(df, str(output_dir))
    logger.info("Saved supplementary synthetic figures.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1_scatter_comparison(results_dir, output_dir)
    fig2_distractor(results_dir, output_dir)
    fig3_efficiency(results_dir, output_dir)
    fig4_ablations(results_dir, output_dir)
    fig_supp_synthetic(results_dir, output_dir)

    logger.info("All paper figures saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir",  default="figures")
    args = parser.parse_args()
    main(args)
