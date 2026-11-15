"""
Scorers: exact match, unbiased pass@k, and math step validity.

Logic step validity (z3) is added together with the logic generators.

Import as:

import logic_slm_scorers as lssco
"""

import fractions
import math
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

import logic_slm_format as lsform

# "a op b = c", optionally prefixed by a description ending in ":".
_MATH_STEP_RE = re.compile(
    r"^(?:.*:\s*)?(-?\d+(?:/\d+)?)\s*([-+*/])\s*(-?\d+(?:/\d+)?)\s*=\s*(-?\d+(?:/\d+)?)$"
)


def exact_match(completion: str, gold: str, task: str) -> bool:
    """
    Whether the answer parsed from `completion` matches `gold`.
    """
    return lsform.answers_match(lsform.parse_answer(completion), gold, task)


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator (Chen et al., 2021).

    :param n: number of samples drawn
    :param c: number of correct samples
    :param k: k <= n
    :return: probability that at least one of k samples (without replacement)
        is correct
    """
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"Invalid n={n}, c={c}, k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _to_frac(s: str) -> fractions.Fraction:
    return fractions.Fraction(s)


def check_math_step(step: str) -> Optional[bool]:
    """
    Check one arithmetic step "a op b = c" exactly.

    :return: True / False if the step parses, None if it is not an
        arithmetic step
    """
    m = _MATH_STEP_RE.match(step.strip())
    if not m:
        return None
    a, op, b, c = _to_frac(m[1]), m[2], _to_frac(m[3]), _to_frac(m[4])
    if op == "+":
        v = a + b
    elif op == "-":
        v = a - b
    elif op == "*":
        v = a * b
    else:
        if b == 0:
            return False
        v = a / b
    return v == c


def math_steps_valid(completion: str) -> bool:
    """
    Whether every step of a math completion is a correct arithmetic step.

    Completions with no steps, or with steps that do not parse, are invalid.
    """
    steps = lsform.parse_steps(completion)
    return bool(steps) and all(check_math_step(s) is True for s in steps)


def score_completions(
    completions: Sequence[str],
    golds: Sequence[str],
    task: str,
) -> Dict[str, float]:
    """
    Aggregate metrics over one completion per problem.

    :return: dict with exact-match accuracy, answer-format rate and, for math,
        step validity and the "correct answer but invalid steps" rate
    """
    if len(completions) != len(golds):
        raise ValueError("completions and golds differ in length")
    n = len(golds)
    em = [exact_match(c, g, task) for c, g in zip(completions, golds)]
    has_ans = [lsform.parse_answer(c) is not None for c in completions]
    out = {
        "n": float(n),
        "exact_match": sum(em) / n,
        "has_answer": sum(has_ans) / n,
    }
    if task == "math":
        valid = [math_steps_valid(c) for c in completions]
        out["steps_valid"] = sum(valid) / n
        out["correct_invalid_steps"] = sum(e and not v for e, v in zip(em, valid)) / n
    return out


def pass_at_k_over_problems(
    correct_counts: List[int], n: int, ks: Sequence[int]
) -> Dict[str, float]:
    """
    Mean pass@k over problems, each with `n` samples.
    """
    return {
        f"pass@{k}": float(np.mean([pass_at_k(n, c, k) for c in correct_counts]))
        for k in ks
        if k <= n
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    """
    Wilson score confidence interval for a proportion.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (centre - half, centre + half)
