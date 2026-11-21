"""
Reference implementation of the output format (see
`references/format_v1.md` and `references/format_v2.md`).

v2 renders exactly like v1; it allows free-text solution lines for real
datasets and makes math answer normalization value-based (decimals, LaTeX
wrappers, currency / percent / unit suffixes).

Every generator renders examples with `render_example()` and every scorer
reads model output with `parse_answer()` / `parse_steps()`.

Import as:

import logic_slm_format as lsform
"""

import dataclasses
import fractions
import re
from typing import Any, Dict, List, Optional

FORMAT_VERSION = "v2"
ANSWER_MARKER = "####"
SOLUTION_HEADER = "Solution:"
TASKS = ("math", "logic")
LOGIC_VERDICTS = ("yes", "no", "unknown")

_FRAC_RE = re.compile(r"^-?\d+/-?\d+$")
# Plain number: integer or decimal, optional thousands separators.
_NUM_RE = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?$")
# Unit words stripped after a number (conservative whitelist; v1 kept
# "12 apples" as is, so object nouns are never stripped).
_UNIT_WORDS = frozenset(
    """
    dollar dollars usd cent cents percent pct
    degree degrees deg
    second seconds sec secs minute minutes min mins hour hours hr hrs
    day days week weeks month months year years
    mm cm m km inch inches in ft foot feet yard yards yd mile miles mi
    meter meters metre metres centimeter centimeters kilometer kilometers
    sq square cubic units unit
    g kg gram grams kilogram kilograms lb lbs pound pounds oz ounce ounces
    l ml liter liters litre litres gallon gallons mph kph
    """.split()
)
_LATEX_SPACES_RE = re.compile(r"\\[,;:! ]|\\quad|\\qquad|~")
_TEXT_CMD_RE = re.compile(r"\\(?:text|textbf|mathrm|mbox|textrm|mathbf|operatorname)\s*\{([^{}]*)\}")
_FRAC_SHORT_RE = re.compile(r"\\frac\s*(\d)\s*(\d)")
_FRAC_LATEX_RE = re.compile(r"^(-?)\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}$")
_VAR_EQ_RE = re.compile(r"^[a-zA-Z]\s*=\s*")
_DEGREE_RE = re.compile(r"\^\s*\{?\\circ\}?")
_PUNCT_SPACE_RE = re.compile(r"\s*([^\w\s\\])\s*")


def _strip_braced_command(s: str, cmd: str) -> str:
    """
    Replace the last `\\cmd{...}` (balanced braces) by its content; if `s`
    contains `\\boxed`, everything outside the box is dropped.
    """
    i = s.rfind(cmd)
    if i < 0:
        return s
    j = i + len(cmd)
    while j < len(s) and s[j] == " ":
        j += 1
    if j >= len(s) or s[j] != "{":
        return s
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1 : k]
    return s


def _strip_latex(a: str) -> str:
    """
    Remove LaTeX wrappers that do not change an answer's meaning.
    """
    for cmd in ("\\boxed", "\\fbox"):
        if cmd in a:
            a = _strip_braced_command(a, cmd)
    a = a.strip()
    for left, right in (("$$", "$$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if a.startswith(left) and a.endswith(right) and len(a) >= len(left) + len(right):
            a = a[len(left) : len(a) - len(right)].strip()
    if a.count("$") == 2 and a.startswith("$") and a.endswith("$") and len(a) > 2:
        a = a[1:-1].strip()
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    a = a.replace("\\left", "").replace("\\right", "")
    a = a.replace("\\%", "%").replace("\\$", "$")
    a = _TEXT_CMD_RE.sub(r"\1", a)
    a = _LATEX_SPACES_RE.sub(" ", a)
    a = _DEGREE_RE.sub("", a)
    a = _FRAC_SHORT_RE.sub(r"\\frac{\1}{\2}", a)
    a = a.replace("\u2212", "-")
    return " ".join(a.split())


def _parse_number(a: str) -> Optional[fractions.Fraction]:
    """
    Parse a numeric answer to an exact value, or None if it is not one.

    Accepts integers, decimals, thousands separators, "a/b", "\\frac{a}{b}",
    a leading "$" (also "-$5"), a trailing "%" or whitelisted unit words, and
    a leading "x = ".
    """
    a = _VAR_EQ_RE.sub("", a).strip()
    a = a.rstrip(".").strip()
    # Trailing unit words (at most two, e.g. "square feet").
    for _ in range(2):
        parts = a.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].lower().rstrip(".") in _UNIT_WORDS:
            a = parts[0].strip()
    a = a.rstrip("%").strip()
    sign = ""
    if a[:1] in "+-":
        sign, a = ("-" if a[0] == "-" else ""), a[1:].strip()
    if a.startswith("$"):
        a = a[1:].strip()
        if a[:1] == "-" and not sign:
            sign, a = "-", a[1:].strip()
    a = sign + a
    if not a or a == "-":
        return None
    try:
        if _NUM_RE.match(a) and any(c.isdigit() for c in a):
            return fractions.Fraction(a.replace(",", ""))
        if _FRAC_RE.match(a):
            num, den = a.split("/")
            if int(den) == 0:
                return None
            return fractions.Fraction(int(num), int(den))
        m = _FRAC_LATEX_RE.match(a)
        if m:
            den = fractions.Fraction(m[3])
            if den == 0:
                return None
            val = fractions.Fraction(m[2]) / den
            return -val if m[1] else val
    except (ValueError, ZeroDivisionError):
        return None
    return None


def format_fraction(f: fractions.Fraction) -> str:
    """
    Canonical string of an exact number: "51", "-7", "7/2".
    """
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


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

    - math (v2): numeric answers compare by exact value and are rendered as
      an integer or a reduced fraction: "051" -> "51", "1,234" -> "1234",
      "6/8" -> "3/4", "3.50" / "7/2" / "\\frac{7}{2}" -> "7/2", "$1,200" ->
      "1200", "25%" -> "25", "12 cm" -> "12", "\\boxed{5}" / "$5$" -> "5".
      Non-numeric answers (e.g. MATH intervals) lose LaTeX wrappers
      (`\\boxed`, `$`, `\\left`/`\\right`, `\\text{}`, `\\dfrac`) and
      spaces next to punctuation: "\\left( 3, \\frac{\\pi}{2} \\right)" ->
      "(3,\\frac{\\pi}{2})". Plain words keep single spaces ("12 apples").
      A zero denominator ("1/0") is kept as is.
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
    # A leading "+" and thousands separators are handled by `_parse_number`
    # (so "+1,234" -> "1234").
    if _FRAC_RE.match(a.lstrip("+")) and int(a.split("/")[1]) == 0:
        return a.lstrip("+")
    a = _strip_latex(a)
    f = _parse_number(a)
    if f is not None:
        return format_fraction(f)
    return _PUNCT_SPACE_RE.sub(r"\1", a)


def answers_match(pred: Optional[str], gold: str, task: str) -> bool:
    """
    Exact match after normalization. A missing prediction never matches.
    """
    p = normalize_answer(pred, task)
    return p is not None and p == normalize_answer(gold, task)
