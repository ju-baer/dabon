#!/usr/bin/env bash
# scripts/run_all_experiments.sh
# --------------------------------
# Master pipeline: runs all three thrusts in sequence.
# Designed for an 8×A100-80GB node with ~2,000 GPU-hours total.
#
# Usage:
#   bash scripts/run_all_experiments.sh [--dry-run] [--tasks gsm8k arena_hard]
#
# Environment variables:
#   MODEL          base model (default: meta-llama/Llama-3.1-8B-Instruct)
#   N_PROMPTS      prompts per task (default: 2000; use 100 for smoke test)
#   DEVICE         torch device  (default: cuda)
#   POOL_SIZE      candidate pool size (default: 128)
#   BUDGET         BoN budget (default: 16)
#   RESULTS_DIR    output root (default: results/)

set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
N_PROMPTS="${N_PROMPTS:-2000}"
DEVICE="${DEVICE:-cuda}"
POOL_SIZE="${POOL_SIZE:-128}"
BUDGET="${BUDGET:-16}"
RESULTS_DIR="${RESULTS_DIR:-results}"
DRY_RUN=false
TASKS="gsm8k arena_hard humaneval creative_writing"

# ---- Argument parsing -------------------------------------------------------
for arg in "$@"; do
  case $arg in
    --dry-run)    DRY_RUN=true ;;
    --tasks=*)    TASKS="${arg#*=}" ;;
    --model=*)    MODEL="${arg#*=}" ;;
    --n_prompts=*) N_PROMPTS="${arg#*=}" ;;
  esac
done

echo "========================================================"
echo " DABoN Experiment Pipeline"
echo "========================================================"
echo " Model:      $MODEL"
echo " Tasks:      $TASKS"
echo " N prompts:  $N_PROMPTS per task"
echo " Pool size:  $POOL_SIZE"
echo " Budget:     $BUDGET"
echo " Results:    $RESULTS_DIR"
echo " Dry run:    $DRY_RUN"
echo "========================================================"

run() {
  echo ""
  echo ">>> $*"
  if [ "$DRY_RUN" = false ]; then
    "$@"
  fi
}

mkdir -p "$RESULTS_DIR"

# =============================================================================
# THRUST I — Negative Result
# =============================================================================
echo ""
echo "===== THRUST I: Correlation Audit ====="

run python -m src.experiments.thrust1_correlation_audit \
    --tasks $TASKS \
    --model "$MODEL" \
    --n_prompts "$N_PROMPTS" \
    --output_dir "$RESULTS_DIR/thrust1/audit" \
    --device "$DEVICE"

echo ""
echo "===== THRUST I: Distractor Injection ====="

run python -m src.experiments.thrust1_distractor_injection \
    --task arena_hard \
    --model "$MODEL" \
    --n_prompts 500 \
    --output_dir "$RESULTS_DIR/thrust1/distractor" \
    --device "$DEVICE"

# =============================================================================
# THRUST II — Mechanism
# =============================================================================
echo ""
echo "===== THRUST II: Synthetic Blind-Manifold ====="

run python -m src.experiments.thrust2_synthetic_manifold \
    --n_sets 2000 \
    --k 16 \
    --m_ensemble 3 \
    --seed 42 \
    --output_dir "$RESULTS_DIR/thrust2/synthetic"

# =============================================================================
# THRUST III — DABoN End-to-End
# =============================================================================
echo ""
echo "===== THRUST III: End-to-End Evaluation ====="

for TASK in $TASKS; do
  run python -m src.experiments.thrust3_dabon_evaluation \
      --task "$TASK" \
      --model "$MODEL" \
      --n_prompts "$N_PROMPTS" \
      --pool_size "$POOL_SIZE" \
      --budgets 2 4 8 16 32 64 \
      --output_dir "$RESULTS_DIR/thrust3/$TASK" \
      --device "$DEVICE"
done

echo ""
echo "===== THRUST III: Ablation Study ====="

run python -m src.experiments.thrust3_ablations \
    --task arena_hard \
    --model "$MODEL" \
    --n_prompts 200 \
    --budget "$BUDGET" \
    --pool_size "$POOL_SIZE" \
    --output_dir "$RESULTS_DIR/thrust3/ablations" \
    --device "$DEVICE"

# =============================================================================
# Aggregate results & generate paper figures
# =============================================================================
echo ""
echo "===== Generating final paper figures ====="

run python scripts/generate_paper_figures.py \
    --results_dir "$RESULTS_DIR" \
    --output_dir figures/

echo ""
echo "========================================================"
echo " All experiments complete!"
echo " Results: $RESULTS_DIR"
echo " Figures: figures/"
echo "========================================================"
