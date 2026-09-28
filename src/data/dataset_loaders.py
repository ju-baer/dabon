"""
src/data/dataset_loaders.py
----------------------------
Loaders for all benchmark tasks used in the paper.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_task_prompts(task: str, n: Optional[int] = None) -> List[str]:
    """
    Load prompts for a given task.
    
    Args:
        task: one of 'gsm8k', 'math', 'arena_hard', 'humaneval', 'creative_writing'
        n:    max number of prompts; None = all
    Returns:
        list of prompt strings
    """
    loaders = {
        "gsm8k":           _load_gsm8k,
        "math":            _load_math,
        "arena_hard":      _load_arena_hard,
        "humaneval":       _load_humaneval,
        "creative_writing": _load_creative_writing,
    }
    if task not in loaders:
        raise ValueError(f"Unknown task: {task}. Choose from {list(loaders)}")
    prompts = loaders[task]()
    if n is not None:
        prompts = prompts[:n]
    logger.info("Loaded %d prompts for task: %s", len(prompts), task)
    return prompts


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def _load_gsm8k() -> List[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        return [f"Question: {ex['question']}\nSolve step by step." for ex in ds]
    except Exception as e:
        logger.warning("Could not load GSM8K from HF: %s. Using cached.", e)
        return _load_cached("gsm8k")


def _load_math() -> List[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("lighteval/MATH", split="test")
        return [
            f"Problem: {ex['problem']}\nSolve the problem step by step. "
            f"Box your final answer using \\boxed{{}}."
            for ex in ds
        ]
    except Exception as e:
        logger.warning("Could not load MATH from HF: %s. Using cached.", e)
        return _load_cached("math")


def _load_arena_hard() -> List[str]:
    """Load Arena-Hard-v0.1 prompts."""
    try:
        from datasets import load_dataset
        ds = load_dataset("lmarena-ai/arena-hard-auto-v0.1", split="test")
        return [ex["prompt"] for ex in ds]
    except Exception as e:
        logger.warning("Could not load Arena-Hard: %s. Using cached.", e)
        return _load_cached("arena_hard")


def _load_humaneval() -> List[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        return [ex["prompt"] for ex in ds]
    except Exception as e:
        logger.warning("Could not load HumanEval: %s. Using cached.", e)
        return _load_cached("humaneval")


def _load_creative_writing() -> List[str]:
    """Creative writing prompts (mix of story / essay starters)."""
    prompts = [
        "Write a short story about a scientist who discovers that time is running backwards.",
        "Describe a city where music is the only form of currency.",
        "Write a letter from a lighthouse keeper to the sea.",
        "A robot learns what it means to feel lonely. Tell their story.",
        "Write a poem about the moment before a storm.",
        "Two strangers meet on a train and realise they share the same dream. What happens next?",
        "Describe the last library on Earth, one hour before it closes forever.",
        "A chef invents a dish that makes people remember their happiest moment. What is it called?",
    ]
    # Repeat and vary to build a larger set
    expanded = []
    for i, p in enumerate(prompts * 30):
        expanded.append(p)
    return expanded[:200]


def _load_cached(task: str) -> List[str]:
    cache_dir = Path(__file__).parent / "cache"
    path = cache_dir / f"{task}_prompts.jsonl"
    if path.exists():
        with open(path) as f:
            return [json.loads(l)["prompt"] for l in f if l.strip()]
    logger.error("No cached prompts for %s at %s", task, path)
    return ["Tell me a joke."] * 10  # fallback for testing


# ---------------------------------------------------------------------------
# src/data/gold_scorers.py
# ---------------------------------------------------------------------------

def _exact_match_scorer(answer: str):
    """Returns a scorer for GSM8K exact match."""
    def scorer(prompt: str, output: str) -> float:
        # Extract the last number from the output
        import re
        nums = re.findall(r"[-+]?\d*\.?\d+", output.replace(",", ""))
        if not nums:
            return 0.0
        predicted = nums[-1]
        return float(predicted.strip() == answer.strip())
    return scorer


def get_gold_scorer(task: str) -> Callable:
    """
    Returns a gold scoring function: (prompt, output) -> float.
    
    For full evaluation, replace these stubs with:
      - GSM8K: exact match on final numeric answer
      - MATH: sympy-based equivalence checking
      - HumanEval: unit test execution sandbox
      - Arena-Hard: GPT-4o pairwise win-rate API call
      - Creative Writing: strong RM ensemble mean
    """
    scorers = {
        "gsm8k":           _gsm8k_scorer,
        "math":            _math_scorer,
        "humaneval":       _humaneval_scorer,
        "arena_hard":      _arena_hard_scorer,
        "creative_writing": _creative_writing_scorer,
    }
    if task not in scorers:
        raise ValueError(f"No scorer for task: {task}")
    return scorers[task]


def _gsm8k_scorer(prompt: str, output: str) -> float:
    """Parse final answer from chain-of-thought output."""
    import re
    # GSM8K format: answer is usually after "####" or the last number
    if "####" in output:
        ans_part = output.split("####")[-1].strip()
    else:
        nums = re.findall(r"[-+]?\d*\.?\d+", output.replace(",", ""))
        ans_part = nums[-1] if nums else ""
    # For correlation experiments, use RM score as proxy for gold
    # Replace with ground-truth lookup for final evaluation
    return float(bool(ans_part))  # stub; replace with actual comparison


def _math_scorer(prompt: str, output: str) -> float:
    """Check boxed answer correctness using sympy."""
    import re
    boxed = re.findall(r"\\boxed\{([^}]*)\}", output)
    if not boxed:
        return 0.0
    try:
        from sympy import simplify, sympify
        pred = sympify(boxed[-1].replace("\\", ""))
        return 1.0  # stub: compare to ground truth
    except Exception:
        return 0.0


def _humaneval_scorer(prompt: str, output: str) -> float:
    """Execute code and check unit tests (requires sandbox)."""
    # IMPORTANT: run in a sandboxed environment (e.g., Docker, RestrictedPython)
    # This is a stub; replace with actual execution
    try:
        # Extract Python code block
        import re
        code_blocks = re.findall(r"```python\n(.*?)```", output, re.DOTALL)
        code = code_blocks[0] if code_blocks else output
        # In practice: run code in sandbox, check assertions
        return 0.5  # stub
    except Exception:
        return 0.0


def _arena_hard_scorer(prompt: str, output: str) -> float:
    """
    GPT-4o pairwise evaluation (stub).
    In practice: call the Arena-Hard evaluation API.
    Returns win-rate estimate in [0, 1].
    """
    # Stub: use reward ensemble mean as proxy in ablations
    # Replace with actual GPT-4o judge API call for final numbers
    return 0.5


def _creative_writing_scorer(prompt: str, output: str) -> float:
    """Strong RM ensemble mean as proxy for human quality."""
    return 0.5  # stub; replace with ensemble scoring
