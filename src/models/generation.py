"""
src/models/generation.py
-------------------------
LLM generation wrapper with vLLM (fast) and HuggingFace (fallback).
Supports all sampling strategies used in the paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    model_name: str
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    # Diverse Beam Search
    num_beams: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    # top-k
    top_k: int = 0
    # vLLM specific
    use_vllm: bool = True
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85


# ---------------------------------------------------------------------------
# Pool generation
# ---------------------------------------------------------------------------

def batch_generate(
    prompts: List[str],
    cfg: GenerationConfig,
    system_prompt: Optional[str] = None,
) -> List[str]:
    """
    Generate one output per prompt.
    Uses vLLM when available, falls back to HuggingFace transformers.
    """
    if cfg.use_vllm:
        try:
            return _vllm_generate(prompts, cfg, system_prompt)
        except ImportError:
            logger.warning("vLLM not available; falling back to HuggingFace.")
    return _hf_generate(prompts, cfg, system_prompt)


def _vllm_generate(
    prompts: List[str],
    cfg: GenerationConfig,
    system_prompt: Optional[str],
) -> List[str]:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=cfg.model_name,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k if cfg.top_k > 0 else -1,
        max_tokens=cfg.max_new_tokens,
    )
    formatted = _format_prompts(prompts, system_prompt, cfg.model_name)
    outputs = llm.generate(formatted, sampling_params)
    return [out.outputs[0].text for out in outputs]


def _hf_generate(
    prompts: List[str],
    cfg: GenerationConfig,
    system_prompt: Optional[str],
) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    formatted = _format_prompts(prompts, system_prompt, cfg.model_name)
    outputs = []
    batch_size = 4

    for i in range(0, len(formatted), batch_size):
        batch = formatted[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=2048).to(model.device)
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        if cfg.num_beam_groups > 1:
            gen_kwargs.update({
                "num_beams": cfg.num_beams,
                "num_beam_groups": cfg.num_beam_groups,
                "diversity_penalty": cfg.diversity_penalty,
                "do_sample": False,
            })
        with torch.no_grad():
            gen_ids = model.generate(**enc, **gen_kwargs)
        input_len = enc["input_ids"].shape[1]
        for ids in gen_ids:
            text = tokenizer.decode(ids[input_len:], skip_special_tokens=True)
            outputs.append(text.strip())

    return outputs


def _format_prompts(
    prompts: List[str],
    system_prompt: Optional[str],
    model_name: str,
) -> List[str]:
    """Apply chat template if model supports it."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        formatted = []
        for p in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": p})
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted.append(text)
            except Exception:
                formatted.append(p)
        return formatted
    except Exception:
        return prompts
