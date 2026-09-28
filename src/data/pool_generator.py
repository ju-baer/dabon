"""
src/data/pool_generator.py
---------------------------
Generates candidate pools for DABoN evaluation.
Uses a mixed-strategy pool:
  50% high-temperature (τ=1.2)
  30% nucleus (p=0.9, τ=0.8)
  20% diverse beam search (λ=0.5)
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from src.models.generation import GenerationConfig, batch_generate

logger = logging.getLogger(__name__)


def generate_pool(
    prompt: str,
    model_name: str,
    pool_size: int = 128,
    device: str = "cuda",
    max_new_tokens: int = 512,
    use_vllm: bool = True,
) -> List[str]:
    """
    Generate a diverse candidate pool for a single prompt.

    Pool composition (Section 3.4A of paper):
      50% high-temperature (τ=1.2)
      30% nucleus sampling  (p=0.9, τ=0.8)
      20% diverse beam search (λ=0.5)

    Args:
        prompt:         The input prompt string.
        model_name:     HuggingFace model identifier.
        pool_size:      Total number of candidates M.
        device:         torch device string.
        max_new_tokens: Max generation length.
        use_vllm:       Prefer vLLM for speed.

    Returns:
        List of pool_size output strings.
    """
    n_high_temp = math.ceil(pool_size * 0.50)
    n_nucleus   = math.ceil(pool_size * 0.30)
    n_dbs       = pool_size - n_high_temp - n_nucleus

    outputs = []

    # --- High temperature ---
    logger.debug("Generating %d high-temp samples", n_high_temp)
    cfg_ht = GenerationConfig(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=1.2,
        top_p=1.0,
        do_sample=True,
        use_vllm=use_vllm,
    )
    outputs += batch_generate([prompt] * n_high_temp, cfg_ht)

    # --- Nucleus sampling ---
    logger.debug("Generating %d nucleus samples", n_nucleus)
    cfg_nuc = GenerationConfig(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        use_vllm=use_vllm,
    )
    outputs += batch_generate([prompt] * n_nucleus, cfg_nuc)

    # --- Diverse Beam Search ---
    logger.debug("Generating %d DBS samples", n_dbs)
    cfg_dbs = GenerationConfig(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        num_beams=max(n_dbs, 4),
        num_beam_groups=max(n_dbs, 4),
        diversity_penalty=0.5,
        do_sample=False,
        use_vllm=False,   # vLLM doesn't support DBS; use HF
    )
    dbs_outputs = batch_generate([prompt] * n_dbs, cfg_dbs)
    outputs += dbs_outputs[:n_dbs]

    assert len(outputs) == pool_size, (
        f"Expected {pool_size} outputs, got {len(outputs)}"
    )
    return outputs


def generate_pool_batch(
    prompts: List[str],
    model_name: str,
    pool_size: int = 128,
    device: str = "cuda",
    max_new_tokens: int = 512,
) -> List[List[str]]:
    """
    Generate pools for a batch of prompts.

    Returns:
        List of lists: one pool per prompt.
    """
    return [
        generate_pool(
            prompt=p,
            model_name=model_name,
            pool_size=pool_size,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        for p in prompts
    ]
