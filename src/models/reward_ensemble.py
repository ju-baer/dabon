"""
src/models/reward_ensemble.py
------------------------------
Wrapper to load and run multiple reward models in parallel, returning
per-model scores and computed disagreement vectors.

Supported models:
  - ArmoRM-Llama3.1-8B-v0.1
  - Skywork-Reward-Gemma-2-27B-v0.2
  - InternLM2-7B-reward
  - Any HuggingFace reward/sequence classification model
  - MC-Dropout fallback for cheap approximation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single reward model wrapper
# ---------------------------------------------------------------------------

@dataclass
class RewardModelConfig:
    model_name: str
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 2048
    dtype: str = "bfloat16"


class SingleRewardModel:
    """Thin wrapper around a HuggingFace reward model."""

    def __init__(self, cfg: RewardModelConfig):
        self.cfg = cfg
        self.dtype = getattr(torch, cfg.dtype)
        logger.info("Loading reward model: %s", cfg.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            torch_dtype=self.dtype,
            device_map=cfg.device,
        )
        self.model.eval()

    @torch.no_grad()
    def score(
        self,
        prompts: List[str],
        outputs: List[str],
    ) -> np.ndarray:
        """
        Score prompt-output pairs.
        
        Args:
            prompts: list of prompt strings.
            outputs: list of corresponding output strings.
        Returns:
            scores: (N,) numpy array of reward scalars.
        """
        assert len(prompts) == len(outputs)
        scores = []
        for i in range(0, len(prompts), self.cfg.batch_size):
            batch_prompts  = prompts[i : i + self.cfg.batch_size]
            batch_outputs  = outputs[i : i + self.cfg.batch_size]
            # Format as chat template if available
            texts = self._format_batch(batch_prompts, batch_outputs)
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
            ).to(self.cfg.device)
            logits = self.model(**enc).logits
            # Most HF reward models output a single logit
            if logits.shape[-1] == 1:
                batch_scores = logits.squeeze(-1).float().cpu().numpy()
            else:
                # Bradley-Terry: take logit for class 1
                batch_scores = logits[:, 1].float().cpu().numpy()
            scores.append(batch_scores)
        return np.concatenate(scores, axis=0)

    def _format_batch(
        self, prompts: List[str], outputs: List[str]
    ) -> List[str]:
        """Format prompt+output for the reward model's expected input."""
        formatted = []
        for p, o in zip(prompts, outputs):
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    messages = [
                        {"role": "user", "content": p},
                        {"role": "assistant", "content": o},
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    formatted.append(text)
                    continue
                except Exception:
                    pass
            formatted.append(f"Human: {p}\nAssistant: {o}")
        return formatted


# ---------------------------------------------------------------------------
# MC-Dropout fallback (cheap ensemble approximation)
# ---------------------------------------------------------------------------

class MCDropoutRewardModel:
    """
    MC-Dropout approximation of ensemble disagreement.
    Enables disagreement estimation from a single model with dropout
    enabled at inference time.
    """

    def __init__(self, cfg: RewardModelConfig, n_samples: int = 10, dropout_p: float = 0.1):
        self.cfg = cfg
        self.n_samples = n_samples
        self.dtype = getattr(torch, cfg.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, torch_dtype=self.dtype, device_map=cfg.device
        )
        # Enable dropout at all layers
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_p
        self.model.train()   # keep dropout active

    @torch.no_grad()
    def score_with_uncertainty(
        self,
        prompts: List[str],
        outputs: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            mean_scores:   (N,)
            score_samples: (N, n_samples) — rows are disagreement vectors
        """
        texts = []
        for p, o in zip(prompts, outputs):
            texts.append(f"Human: {p}\nAssistant: {o}")
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        ).to(self.cfg.device)

        all_scores = []
        for _ in range(self.n_samples):
            logits = self.model(**enc).logits
            scores = logits.squeeze(-1).float().cpu().numpy()
            all_scores.append(scores)

        score_matrix = np.stack(all_scores, axis=1)   # (N, n_samples)
        return score_matrix.mean(axis=1), score_matrix


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------

class RewardEnsemble:
    """
    Runs multiple reward models and returns:
      - per-model scores: (N, m)
      - mean scores:      (N,)
      - disagreement vectors: (N, m)
    """

    def __init__(self, model_configs: List[RewardModelConfig]):
        self.models = [SingleRewardModel(cfg) for cfg in model_configs]
        logger.info("Ensemble of %d reward models loaded.", len(self.models))

    @property
    def m(self) -> int:
        return len(self.models)

    def score_all(
        self,
        prompts: List[str],
        outputs: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Score all prompt-output pairs with all ensemble members.
        
        Returns dict with:
          'per_model_scores':      (N, m)
          'mean_scores':           (N,)
          'disagreement_vectors':  (N, m)
        """
        per_model = np.stack(
            [m.score(prompts, outputs) for m in self.models], axis=1
        )   # (N, m)
        mean_scores = per_model.mean(axis=1)
        disagreement = per_model - mean_scores[:, np.newaxis]
        return {
            "per_model_scores":     per_model,
            "mean_scores":          mean_scores,
            "disagreement_vectors": disagreement,
        }


# ---------------------------------------------------------------------------
# Factory: build ensemble from named configs
# ---------------------------------------------------------------------------

KNOWN_MODELS = {
    "armo": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    "skywork": "Skywork/Skywork-Reward-Gemma-2-27B-v0.2",
    "internlm": "internlm/internlm2-7b-reward",
}


def build_ensemble(
    model_keys: List[str],
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> RewardEnsemble:
    """
    Args:
        model_keys: list of keys from KNOWN_MODELS or full HF model names.
    """
    configs = []
    for key in model_keys:
        name = KNOWN_MODELS.get(key, key)
        configs.append(RewardModelConfig(model_name=name, device=device, dtype=dtype))
    return RewardEnsemble(configs)
