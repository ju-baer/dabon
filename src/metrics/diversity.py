"""
src/metrics/diversity.py
------------------------
All diversity metrics used in the paper:
  - Geometric: APCD (cosine), MMD, Self-BLEU / Distinct-N
  - Reward-aware: reward variance, reward range
  - Proposed: Reward-Nullspace Diversity (RND)
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Geometric metrics
# ---------------------------------------------------------------------------

def average_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
    """
    APCD: mean cosine distance over all pairs in the set.
    
    Args:
        embeddings: (K, d) array of unit-normalised embeddings.
    Returns:
        Scalar in [0, 2].
    """
    K = len(embeddings)
    if K < 2:
        return 0.0
    # normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed = embeddings / norms
    # cosine similarity matrix
    sim = normed @ normed.T          # (K, K)
    # upper-triangle only, excluding diagonal
    i, j = np.triu_indices(K, k=1)
    distances = 1.0 - sim[i, j]
    return float(distances.mean())


def mmd_rbf(
    embeddings: np.ndarray,
    reference: np.ndarray,
    bandwidth: Optional[float] = None,
) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel between sample set and reference.
    
    Args:
        embeddings: (K, d) sample set embeddings.
        reference:  (n, d) reference set embeddings.
        bandwidth:  RBF bandwidth σ²; if None, uses median heuristic.
    Returns:
        Scalar MMD² estimate.
    """
    K, n = len(embeddings), len(reference)
    if bandwidth is None:
        all_emb = np.vstack([embeddings, reference])
        dists = cdist(all_emb, all_emb, "sqeuclidean")
        bandwidth = float(np.median(dists[dists > 0]))

    def rbf(X, Y):
        d = cdist(X, Y, "sqeuclidean")
        return np.exp(-d / (2 * bandwidth))

    kxx = rbf(embeddings, embeddings)
    kyy = rbf(reference, reference)
    kxy = rbf(embeddings, reference)

    mmd2 = (kxx.sum() / (K * K)
            + kyy.sum() / (n * n)
            - 2 * kxy.sum() / (K * n))
    return float(max(mmd2, 0.0))


def self_bleu(texts: List[str], max_ngram: int = 4) -> float:
    """
    Self-BLEU: mean BLEU of each text treated as hypothesis against
    all others as references. Lower = more diverse.
    
    Returns lexical diversity = 1 - Self-BLEU.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download("punkt", quiet=True)
    except ImportError:
        raise ImportError("Run: pip install nltk")

    smoother = SmoothingFunction().method1
    tokenised = [t.lower().split() for t in texts]
    bleu_scores = []
    for i, hyp in enumerate(tokenised):
        refs = [t for j, t in enumerate(tokenised) if j != i]
        if not refs:
            continue
        score = sentence_bleu(refs, hyp, smoothing_function=smoother)
        bleu_scores.append(score)
    mean_self_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    return 1.0 - mean_self_bleu  # lexical diversity


def distinct_n(texts: List[str], n: int = 4) -> float:
    """
    Distinct-N: ratio of unique n-grams to total n-grams across all texts.
    """
    from collections import Counter

    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        all_ngrams.extend(
            tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
        )
    if not all_ngrams:
        return 0.0
    unique = len(set(all_ngrams))
    total = len(all_ngrams)
    return unique / total


# ---------------------------------------------------------------------------
# Reward-aware metrics
# ---------------------------------------------------------------------------

def reward_variance(rewards: np.ndarray) -> float:
    return float(np.std(rewards))


def reward_range(rewards: np.ndarray) -> float:
    return float(rewards.max() - rewards.min()) if len(rewards) > 1 else 0.0


# ---------------------------------------------------------------------------
# Proposed: Reward-Nullspace Diversity (RND)
# ---------------------------------------------------------------------------

def compute_disagreement_vectors(
    reward_scores: np.ndarray,
) -> np.ndarray:
    """
    Compute disagreement vectors Δ(y) ∈ ℝᵐ for each output.
    
    Args:
        reward_scores: (K, m) array where reward_scores[i, j] = R̂_j(y_i).
    Returns:
        disagreement: (K, m) array, each row centred to zero mean.
    """
    mean_reward = reward_scores.mean(axis=1, keepdims=True)  # (K, 1)
    return reward_scores - mean_reward                        # (K, m)


def rnd(
    reward_scores: np.ndarray,
    lam: Optional[float] = None,
) -> float:
    """
    Reward-Nullspace Diversity (RND): log-det of the disagreement Gram matrix.
    
    RND(S) = log det(K_Δ + λ I_K)
    where K_Δ[i,j] = <Δ(y_i), Δ(y_j)>
    
    Args:
        reward_scores: (K, m) array of ensemble reward scores.
        lam: regularisation; if None uses adaptive 1e-4 * tr(K) / K.
    Returns:
        Scalar log-det value.
    """
    delta = compute_disagreement_vectors(reward_scores)   # (K, m)
    K_mat = delta @ delta.T                               # (K, K) Gram matrix

    K = len(reward_scores)
    if lam is None:
        trace = np.trace(K_mat)
        lam = max(1e-8, 1e-4 * trace / K)

    reg = K_mat + lam * np.eye(K)
    # Numerically stable log-det via Cholesky
    try:
        L = np.linalg.cholesky(reg)
        return float(2.0 * np.sum(np.log(np.diag(L))))
    except np.linalg.LinAlgError:
        # Fall back to eigenvalue decomposition
        eigvals = np.linalg.eigvalsh(reg)
        eigvals = np.maximum(eigvals, 1e-15)
        return float(np.sum(np.log(eigvals)))


# ---------------------------------------------------------------------------
# Gradient-conditional decomposition (Section 4.3)
# ---------------------------------------------------------------------------

def gradient_conditional_decomposition(
    embeddings: np.ndarray,
    reward_gradient: np.ndarray,
) -> dict:
    """
    Decompose pairwise distances into reward-parallel and reward-nullspace
    components.
    
    Args:
        embeddings:      (K, d) output embeddings.
        reward_gradient: (d,) local reward gradient in embedding space.
    Returns:
        dict with keys 'parallel_diversity', 'nullspace_diversity'.
    """
    u = reward_gradient / (np.linalg.norm(reward_gradient) + 1e-12)
    K = len(embeddings)
    parallel_dists, null_dists = [], []

    for i in range(K):
        for j in range(i + 1, K):
            diff = embeddings[i] - embeddings[j]
            proj = np.dot(diff, u)
            parallel_component = proj * u
            null_component = diff - parallel_component
            parallel_dists.append(abs(proj))
            null_dists.append(np.linalg.norm(null_component))

    return {
        "parallel_diversity": float(np.mean(parallel_dists)),
        "nullspace_diversity": float(np.mean(null_dists)),
    }


# ---------------------------------------------------------------------------
# Convenience: compute all metrics for a sample set
# ---------------------------------------------------------------------------

def compute_all_metrics(
    texts: List[str],
    embeddings: np.ndarray,
    reward_scores: np.ndarray,
    reference_embeddings: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all diversity metrics for a single sample set.
    
    Args:
        texts:                 K output texts.
        embeddings:            (K, d) SBERT embeddings.
        reward_scores:         (K, m) ensemble reward scores.
        reference_embeddings:  (n, d) reference pool for MMD; optional.
    Returns:
        dict mapping metric name → scalar value.
    """
    rewards_mean = reward_scores.mean(axis=1)  # (K,) mean reward

    metrics = {
        "apcd":          average_pairwise_cosine_distance(embeddings),
        "self_bleu":     self_bleu(texts),
        "distinct_4":    distinct_n(texts, n=4),
        "reward_var":    reward_variance(rewards_mean),
        "reward_range":  reward_range(rewards_mean),
        "rnd":           rnd(reward_scores),
        "max_reward":    float(rewards_mean.max()),
        "mean_reward":   float(rewards_mean.mean()),
    }

    if reference_embeddings is not None:
        metrics["mmd"] = mmd_rbf(embeddings, reference_embeddings)

    return metrics
