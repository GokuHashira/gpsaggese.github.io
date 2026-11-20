"""
Reference implementation of the frozen output format (see
`references/format_v1.md`).

Every generator renders examples with `render_example()` and every scorer
reads model output with `parse_answer()` / `parse_steps()`.

Import as:

import logic_slm_format as lsform
"""

import dataclasses
import fractions
import re
from typing import Any, Dict, List, Optional

FORMAT_VERSION = "v1"
ANSWER_MARKER = "####"
SOLUTION_HEADER = "Solution:"
TASKS = ("math", "logic")
LOGIC_VERDICTS = ("yes", "no", "unknown")

_INT_RE = re.compile(r"^-?\d+$")
_FRAC_RE = re.compile(r"^-?\d+/-?\d+$")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+$")


@dataclasses.dataclass(frozen=True)
class Example:
    """
    One problem with its worked solution.

    :param task: "math" or "logic"
    :param problem: math problem text, or the logic premises without the
        "Premises:" prefix (e.g. "p -> q. p.")
    :param steps: worked solution steps, one per line
    :param answer: final answer (canonical form)
    :param question: logic question formula without the trailing "?"
    :param meta: generator metadata (level, number of steps, seed, ...); not
        rendered
    """

    task: str
    problem: str
    steps: List[str]
    answer: str
    question: Optional[str] = None
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"Unknown task '{self.task}'")
        if self.task == "logic":
            if self.question is None:
                raise ValueError("Logic examples need a question")
            if self.answer not in LOGIC_VERDICTS:
                raise ValueError(f"Invalid logic verdict '{self.answer}'")
        for s in self.steps:
            if "\n" in s or not s.strip():
                raise ValueError(f"Steps must be single non-empty lines: {s!r}")
            if s.strip().startswith(ANSWER_MARKER):
                raise ValueError(f"Step looks like an answer line: {s!r}")
        if "\n" in self.answer or not self.answer.strip():
            raise ValueError(f"Invalid answer {self.answer!r}")


# #############################################################################
# Rendering
# #############################################################################


def render_prompt(ex: Example) -> str:
    """
    Render the prompt, ending with "Solution:\\n".
    """
    if ex.task == "math":
        head = f"Problem: {ex.problem}\n"
    else:
        head = f"Premises: {ex.problem}\nQuestion: {ex.question}?\n"
    return f"{head}{SOLUTION_HEADER}\n"


def render_solution(ex: Example) -> str:
    """
    Render the solution: step lines followed by the answer line.

    The solution has no trailing newline; the tokenizer appends EOS.
    """
    lines = list(ex.steps) + [f"{ANSWER_MARKER} {ex.answer}"]
    return "\n".join(lines)


def render_example(ex: Example) -> str:
    """
    Render the full training text (prompt + solution).
    """
    return render_prompt(ex) + render_solution(ex)


# #############################################################################
# Parsing
# #############################################################################


def _solution_part(text: str) -> str:
    """
    Return the text after the first "Solution:" line, or the whole text if
    there is none (a bare model completion).
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == SOLUTION_HEADER:
            return "\n".join(lines[i + 1 :])
    return text


def _answer_line_index(lines: List[str]) -> Optional[int]:
    for i, line in enumerate(lines):
        if line.strip().startswith(ANSWER_MARKER):
            return i
    return None


def parse_answer(text: str) -> Optional[str]:
    """
    Extract the raw answer from the first "####" line of the solution.

    :return: the stripped answer, or None if there is no answer line or it
        is empty
    """
    lines = _solution_part(text).split("\n")
    idx = _answer_line_index(lines)
    if idx is None:
        return None
    ans = lines[idx].strip()[len(ANSWER_MARKER) :].strip()
    return ans or None


def parse_steps(text: str) -> List[str]:
    """
    Extract the non-empty step lines before the answer line.

    If there is no answer line, all non-empty solution lines are returned.
    """
    lines = _solution_part(text).split("\n")
    idx = _answer_line_index(lines)
    if idx is not None:
        lines = lines[:idx]
    return [line.strip() for line in lines if line.strip()]


def normalize_answer(answer: Optional[str], task: str) -> Optional[str]:
    """
    Canonicalize an answer for exact-match comparison.

    - math: integers lose leading zeros, "+" and "-0"; "1,234" -> "1234";
      fractions are reduced ("6/8" -> "3/4", "4/2" -> "2"); anything else is
      only whitespace-collapsed.
    - logic: lowercased, trailing "." removed.

    :return: canonical answer, or None if `answer` is None
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'")
    if answer is None:
        return None
    a = " ".join(answer.split())
    if task == "logic":
        return a.lower().rstrip(".")
    # Strip "+" first, so that "+1,234" is also recognized as "1,234".
    a = a.lstrip("+")
    if _THOUSANDS_RE.match(a):
        a = a.replace(",", "")
    if _INT_RE.match(a):
        return str(int(a))
    if _FRAC_RE.match(a):
        num, den = a.split("/")
        if int(den) == 0:
            return a
        f = fractions.Fraction(int(num), int(den))
        if f.denominator == 1:
            return str(f.numerator)
        return f"{f.numerator}/{f.denominator}"
    return a


def answers_match(pred: Optional[str], gold: str, task: str) -> bool:
    """
    Exact match after normalization. A missing prediction never matches.
    """
    p = normalize_answer(pred, task)
    return p is not None and p == normalize_answer(gold, task)
