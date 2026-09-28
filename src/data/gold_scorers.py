"""
src/data/gold_scorers.py
-------------------------
Gold-quality scoring functions for each benchmark task.

Each scorer has the signature:
    scorer(prompt: str, output: str) -> float

Values are in [0, 1] (or higher for tasks with partial credit).
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def _parse_final_number(text: str) -> Optional[str]:
    """Extract the final numeric answer from chain-of-thought text."""
    # Check for #### marker (GSM8K official format)
    if "####" in text:
        part = text.split("####")[-1].strip()
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", part)
        if nums:
            return nums[0].replace(",", "")
    # Fallback: last number in the text
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def make_gsm8k_scorer(gold_answers: dict) -> Callable:
    """
    Args:
        gold_answers: dict mapping prompt -> gold numeric answer string
    """
    def scorer(prompt: str, output: str) -> float:
        predicted = _parse_final_number(output)
        gold      = gold_answers.get(prompt)
        if gold is None or predicted is None:
            return 0.0
        try:
            return float(abs(float(predicted) - float(gold)) < 1e-3)
        except ValueError:
            return float(predicted.strip() == str(gold).strip())
    return scorer


# ---------------------------------------------------------------------------
# MATH (sympy-based)
# ---------------------------------------------------------------------------

def make_math_scorer(gold_answers: dict) -> Callable:
    """
    Uses sympy to check symbolic equivalence of boxed answers.
    """
    def scorer(prompt: str, output: str) -> float:
        boxes = re.findall(r"\\boxed\{([^}]*)\}", output)
        if not boxes:
            return 0.0
        predicted_str = boxes[-1].strip()
        gold_str      = gold_answers.get(prompt, "")
        if not gold_str:
            return 0.0
        try:
            from sympy import simplify, sympify, N
            pred_val = sympify(predicted_str.replace("\\", "").replace("^", "**"))
            gold_val = sympify(gold_str.replace("\\", "").replace("^", "**"))
            diff     = simplify(pred_val - gold_val)
            return float(abs(complex(N(diff))) < 1e-6)
        except Exception:
            return float(predicted_str.strip() == gold_str.strip())
    return scorer


# ---------------------------------------------------------------------------
# HumanEval (code execution)
# ---------------------------------------------------------------------------

def make_humaneval_scorer(
    test_cases: dict,
    timeout: float = 5.0,
) -> Callable:
    """
    Execute generated code against unit tests in a restricted environment.

    IMPORTANT: Always run in a sandboxed container in production.

    Args:
        test_cases: dict mapping prompt -> test_code string
        timeout:    per-test execution timeout in seconds
    """
    def scorer(prompt: str, output: str) -> float:
        test_code = test_cases.get(prompt, "")
        if not test_code:
            return 0.0
        # Extract Python code block
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", output, re.DOTALL)
        code = code_blocks[0] if code_blocks else output

        full_code = code + "\n\n" + test_code
        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout))
            exec_globals: dict = {}
            exec(compile(full_code, "<string>", "exec"), exec_globals)  # noqa: S102
            signal.alarm(0)
            return 1.0
        except TimeoutError:
            return 0.0
        except Exception as e:
            logger.debug("HumanEval execution error: %s", e)
            return 0.0

    return scorer


# ---------------------------------------------------------------------------
# Arena-Hard (GPT-4o judge)
# ---------------------------------------------------------------------------

def make_arena_hard_scorer(
    reference_answers: dict,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o",
) -> Callable:
    """
    Pairwise GPT-4o judge: compares output to reference answer.
    Returns win probability in [0, 1].

    For ablations / fast experiments, uses reward ensemble mean as proxy
    when api_key is not provided.
    """
    import os
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    JUDGE_PROMPT = (
        "You are a judge evaluating two responses to a user query.\n\n"
        "Query: {prompt}\n\n"
        "Response A: {ref}\n\n"
        "Response B: {output}\n\n"
        "Which response is better? Reply with exactly one of: 'A', 'B', or 'tie'."
    )

    def scorer(prompt: str, output: str) -> float:
        ref = reference_answers.get(prompt, "")
        if not ref or api_key is None:
            # Fallback: length heuristic (useful for smoke testing only)
            return min(len(output) / max(len(ref) + 1, 1), 2.0) / 2.0

        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        prompt=prompt, ref=ref, output=output
                    ),
                }],
                temperature=0.0,
                max_tokens=5,
            )
            verdict = resp.choices[0].message.content.strip().upper()
            if verdict == "B":
                return 1.0
            elif verdict == "A":
                return 0.0
            else:
                return 0.5
        except Exception as e:
            logger.warning("GPT-4o judge error: %s", e)
            return 0.5

    return scorer


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_gold_scorer(task: str) -> Callable:
    """
    Return a simple scorer for quick experiments.
    For proper evaluation, use make_*_scorer with loaded ground-truth data.
    """
    def stub_scorer(prompt: str, output: str) -> float:
        """Placeholder: returns output length normalised to [0,1]."""
        return min(len(output.split()) / 200.0, 1.0)

    task_map = {
        "gsm8k":            stub_scorer,
        "math":             stub_scorer,
        "humaneval":        stub_scorer,
        "arena_hard":       stub_scorer,
        "creative_writing": stub_scorer,
    }
    scorer = task_map.get(task)
    if scorer is None:
        raise ValueError(f"Unknown task: {task}")
    logger.warning(
        "Using stub scorer for task '%s'. "
        "Replace with make_%s_scorer() for real evaluation.", task, task
    )
    return scorer
