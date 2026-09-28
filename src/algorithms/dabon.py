"""
src/algorithms/dabon.py
-----------------------
Disagreement-Aware Best-of-N (DABoN):
  - Greedy submodular maximisation of a DPP disagreement kernel
    subject to an exploitation-quality (mean reward) constraint.
  - Achieves (1 - 1/e) approximation ratio (see Theorem 5 in paper).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DABoNConfig:
    budget: int = 16                    # N: number of outputs to select
    pool_size: int = 128                # M: initial candidate pool (M >> N)
    beta: float = 1.0                   # exploitation/diversity tradeoff
    adaptive_beta: bool = True          # scale beta by reward confidence
    sigma: Optional[float] = None       # RBF bandwidth; None = median heuristic
    lambda_reg: Optional[float] = None  # kernel regularisation; None = adaptive
    temperature: float = 1.0            # pool sampling temperature
    pool_temperatures: List[float] = field(
        default_factory=lambda: [1.2, 0.8, 1.0]
    )
    pool_fractions: List[float] = field(
        default_factory=lambda: [0.5, 0.3, 0.2]
    )
    top_k_candidates: int = 50          # speed: only evaluate top-K by reward


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _rbf_kernel(
    delta_i: np.ndarray,
    delta_j: np.ndarray,
    sigma: float,
) -> float:
    diff = delta_i - delta_j
    return float(np.exp(-np.dot(diff, diff) / (2 * sigma ** 2)))


def _build_kernel_matrix(
    deltas: np.ndarray,
    sigma: float,
    lam: float,
) -> np.ndarray:
    """Build the full (K, K) regularised kernel matrix."""
    K = len(deltas)
    mat = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            mat[i, j] = _rbf_kernel(deltas[i], deltas[j], sigma)
    mat += lam * np.eye(K)
    return mat


def _marginal_gain_rank1(
    k_yy: float,
    k_yS: np.ndarray,
    K_S_inv: np.ndarray,
) -> float:
    """
    Efficient rank-1 marginal log-det gain via the matrix determinant lemma:
        log det(K_{S ∪ {y}}) - log det(K_S)
        = log(k(y,y) - k(y,S) K_S^{-1} k(S,y))
    
    Args:
        k_yy:    scalar k(y, y).
        k_yS:    (|S|,) vector k(y, s_i) for s_i in S.
        K_S_inv: (|S|, |S|) inverse of current kernel matrix.
    Returns:
        Scalar marginal gain.
    """
    schur = k_yy - k_yS @ K_S_inv @ k_yS
    schur = max(schur, 1e-12)
    return float(np.log(schur))


def _sherman_morrison_update(
    K_inv: np.ndarray,
    k_yS: np.ndarray,
    k_yy: float,
    schur: float,
) -> np.ndarray:
    """
    Extend K_S^{-1} to K_{S∪{y}}^{-1} via block-matrix inversion.
    O(|S|²) per step.
    """
    n = len(K_inv)
    # Build new inverse block by block
    u = K_inv @ k_yS                    # (n,)
    s_inv = 1.0 / schur
    new_inv = np.zeros((n + 1, n + 1))
    new_inv[:n, :n] = K_inv + s_inv * np.outer(u, u)
    new_inv[:n, n] = -s_inv * u
    new_inv[n, :n] = -s_inv * u
    new_inv[n, n] = s_inv
    return new_inv


# ---------------------------------------------------------------------------
# Adaptive beta
# ---------------------------------------------------------------------------

class BetaScheduler:
    """Track historical max rewards and compute adaptive beta."""

    def __init__(self, beta0: float, percentile: float = 90.0):
        self.beta0 = beta0
        self.percentile = percentile
        self._history: List[float] = []

    def update(self, max_reward: float) -> None:
        self._history.append(max_reward)

    def get_beta(self, current_max: float) -> float:
        if not self._history:
            return self.beta0
        hist_val = float(np.percentile(self._history, self.percentile))
        confidence = current_max / (hist_val + 1e-8)
        beta = self.beta0 * (1.0 - min(confidence, 1.0))
        return max(beta, 0.0)


# ---------------------------------------------------------------------------
# Core DABoN selector
# ---------------------------------------------------------------------------

class DABoNSelector:
    """
    Greedy submodular selector.

    Usage::
        selector = DABoNSelector(cfg)
        indices = selector.select(mean_rewards, disagreement_vectors)
    """

    def __init__(self, cfg: DABoNConfig):
        self.cfg = cfg
        self.beta_scheduler = BetaScheduler(cfg.beta) if cfg.adaptive_beta else None

    # ------------------------------------------------------------------
    def _compute_sigma(self, deltas: np.ndarray) -> float:
        if self.cfg.sigma is not None:
            return self.cfg.sigma
        K = len(deltas)
        dists = []
        for i in range(K):
            for j in range(i + 1, K):
                d = deltas[i] - deltas[j]
                dists.append(np.dot(d, d) ** 0.5)
        return float(np.median(dists)) if dists else 1.0

    def _compute_lambda(self, K_mat: np.ndarray, K: int) -> float:
        if self.cfg.lambda_reg is not None:
            return self.cfg.lambda_reg
        trace = np.trace(K_mat)
        return max(1e-8, 1e-4 * trace / K)

    # ------------------------------------------------------------------
    def select(
        self,
        mean_rewards: np.ndarray,
        disagreement_vectors: np.ndarray,
        beta_override: Optional[float] = None,
    ) -> List[int]:
        """
        Select N indices from the candidate pool.
        
        Args:
            mean_rewards:          (M,) mean reward for each candidate.
            disagreement_vectors:  (M, m) Δ(y) vectors.
            beta_override:         if set, overrides adaptive beta for this call.
        Returns:
            selected_indices: list of N integer indices into the pool.
        """
        M = len(mean_rewards)
        N = min(self.cfg.budget, M)

        sigma = self._compute_sigma(disagreement_vectors)

        # Pre-compute full kernel row cache
        # k_cache[i, j] = k(y_i, y_j)
        k_cache = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                k_cache[i, j] = _rbf_kernel(
                    disagreement_vectors[i], disagreement_vectors[j], sigma
                )

        # Determine beta
        if beta_override is not None:
            beta = beta_override
        elif self.cfg.adaptive_beta and self.beta_scheduler is not None:
            cur_max = float(mean_rewards.max())
            beta = self.beta_scheduler.get_beta(cur_max)
            self.beta_scheduler.update(cur_max)
        else:
            beta = self.cfg.beta

        # Pre-select top-K candidates by reward to speed up inner loop
        top_k = min(self.cfg.top_k_candidates, M)
        top_indices = np.argsort(mean_rewards)[-top_k:][::-1].tolist()
        candidate_set = set(top_indices)

        selected: List[int] = []
        K_inv: Optional[np.ndarray] = None   # running inverse

        for step in range(N):
            best_idx, best_score = -1, -float("inf")
            best_schur = None

            remaining = candidate_set - set(selected)
            if not remaining:
                logger.warning("Pool exhausted at step %d", step)
                break

            for idx in remaining:
                if step == 0:
                    # First selection: log det([k(y,y) + lam]) = log(k_yy + lam)
                    lam = self._compute_lambda(
                        np.array([[k_cache[idx, idx]]]), 1
                    )
                    gain = float(np.log(k_cache[idx, idx] + lam))
                else:
                    k_yS = np.array([k_cache[idx, s] for s in selected])
                    k_yy = k_cache[idx, idx]
                    gain = _marginal_gain_rank1(k_yy, k_yS, K_inv)

                score = mean_rewards[idx] + beta * gain
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    if step > 0:
                        k_yS_best = np.array([k_cache[best_idx, s] for s in selected])
                        schur_best = (
                            k_cache[best_idx, best_idx]
                            - k_yS_best @ K_inv @ k_yS_best
                        )
                        best_schur = max(schur_best, 1e-12)

            selected.append(best_idx)

            # Update inverse
            if step == 0:
                lam = self._compute_lambda(
                    np.array([[k_cache[best_idx, best_idx]]]), 1
                )
                K_inv = np.array([[1.0 / (k_cache[best_idx, best_idx] + lam)]])
            else:
                k_yS_best = np.array([k_cache[best_idx, s] for s in selected[:-1]])
                K_inv = _sherman_morrison_update(
                    K_inv, k_yS_best, k_cache[best_idx, best_idx], best_schur
                )

        logger.debug("DABoN selected %d outputs, final beta=%.3f", len(selected), beta)
        return selected

    # ------------------------------------------------------------------
    def select_best(
        self,
        mean_rewards: np.ndarray,
        disagreement_vectors: np.ndarray,
        outputs: List,
    ):
        """
        Convenience wrapper: returns the best output from the selected set.
        """
        indices = self.select(mean_rewards, disagreement_vectors)
        if not indices:
            return outputs[int(np.argmax(mean_rewards))]
        # Return highest mean-reward among selected
        best_local = int(np.argmax(mean_rewards[np.array(indices)]))
        return outputs[indices[best_local]]


# ---------------------------------------------------------------------------
# Standard Best-of-N baseline (for fair comparison)
# ---------------------------------------------------------------------------

def standard_bon(
    mean_rewards: np.ndarray,
    outputs: List,
    N: int,
) -> tuple:
    """Select the best output from N iid samples by mean reward."""
    if N >= len(outputs):
        best_idx = int(np.argmax(mean_rewards))
        return outputs[best_idx], best_idx
    # Sample N without replacement from pool
    perm = np.random.permutation(len(outputs))[:N]
    local_best = int(np.argmax(mean_rewards[perm]))
    best_idx = int(perm[local_best])
    return outputs[best_idx], best_idx
