"""
M2 generator: symbolic math (linear equations, function evaluation and
composition, comparisons) with worked steps.

In the style of DeepMind `mathematics_dataset` (own implementation). Values
may be negative integers or exact fractions (`3/4`). Every solution step is
one operation `<description>: a op b = c` or `a op b = c` (see
`references/format_v1.md`, section 2).

Problem kinds (meta field `kind`):
- `linear`: `Solve 3*x + 5 = 20 for x.`; x terms and constants may appear on
  both sides. Steps combine the x coefficients, then the constants, then
  divide. The solution is an integer or a reduced fraction.
- `eval`: `Let f(x) = 2*x^2 - 3*x + 1. What is f(4)?`
- `compose`: `Let f(x) = ... and g(x) = .... What is f(g(2))?`
- `compare`: `Which is larger: 3/4 or 5/7?` (or the largest / smallest of
  up to n+1 values). Steps cross-multiply and subtract; the answer is one of
  the given values, so the last step value is a difference, not the answer.

Import as:

import logic_slm_math_m2 as lsmam2
"""

import argparse
import fractions
import json
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import logic_slm_format as lsform
import logic_slm_math_m1 as lsmam1

Fraction = fractions.Fraction
KINDS = ("linear", "eval", "compose", "compare")
_VARIABLES = ("x", "y", "z", "t", "n", "m", "k", "a", "b", "c", "p", "w")
_FUNC_NAMES = ("f", "g", "h", "p", "q", "r", "s", "u", "v")
_MAX_ATTEMPTS = 10_000


def fmt_number(v: Fraction) -> str:
    """
    Format an exact number in canonical form (`5`, `-7`, `3/4`, `-3/4`).
    """
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    return f"{v.numerator}/{v.denominator}"


def _step(a: Fraction, op: str, b: Fraction, desc: str = "") -> Tuple[str, Fraction]:
    """
    Compute one operation and render its step line.

    :return: (step line, result)
    """
    a, b = Fraction(a), Fraction(b)
    if op == "+":
        c = a + b
    elif op == "-":
        c = a - b
    elif op == "*":
        c = a * b
    else:
        c = a / b
    line = f"{fmt_number(a)} {op} {fmt_number(b)} = {fmt_number(c)}"
    if desc:
        line = f"{desc}: {line}"
    return line, c


def _nonzero(rng: random.Random, lo: int, hi: int) -> int:
    """
    Sample a non-zero integer with |v| in [lo, hi] and a random sign.
    """
    return rng.choice((-1, 1)) * rng.randint(lo, hi)


# #############################################################################
# Linear equations
# #############################################################################


def _render_terms(terms: List[Tuple[int, bool]], var: str) -> str:
    """
    Render signed terms (coefficient, is_x_term) as `3*x - 5 + x`.
    """
    out = []
    for i, (c, is_x) in enumerate(terms):
        mag = abs(c)
        body = (var if mag == 1 else f"{mag}*{var}") if is_x else str(mag)
        if i == 0:
            out.append(f"-{body}" if c < 0 else body)
        else:
            out.append(f"{'-' if c < 0 else '+'} {body}")
    return " ".join(out)


def _split_sum(rng: random.Random, total: int, k: int, bound: int) -> Optional[List[int]]:
    """
    Split `total` into `k` non-zero integers with |v| <= bound.

    :return: the parts, or None on failure
    """
    parts = [_nonzero(rng, 1, bound) for _ in range(k - 1)]
    last = total - sum(parts)
    if last == 0 or abs(last) > bound:
        return None
    parts.insert(rng.randrange(k), last)
    return parts


def _gen_linear(
    rng: random.Random, n_steps: int, max_digits: int, frac_prob: float
) -> Optional[lsform.Example]:
    max_const = 10**max_digits - 1
    max_coef = min(12, max_const)
    var = rng.choice(_VARIABLES)
    divide = n_steps == 1 and rng.random() < 0.7 or n_steps > 1 and rng.random() < 0.9
    extra = n_steps - int(divide)
    # Distribute the extra steps among: x terms on the left (beyond the
    # first), x terms on the right, constants on the right (beyond the
    # first), constants on the left.
    cuts = sorted(rng.randint(0, extra) for _ in range(3))
    a, b, c, d = cuts[0], cuts[1] - cuts[0], cuts[2] - cuts[1], extra - cuts[2]
    if not divide and c + d == 0:
        return None
    # Coefficient A and constant B with A * x = B.
    if divide:
        coef = _nonzero(rng, 1, max_coef)
        if coef == 1:
            return None
    else:
        coef = 1
    if divide and rng.random() < frac_prob:
        const = _nonzero(rng, 1, max_const)
        if const % coef == 0:
            return None
    else:
        lim = max_const // abs(coef)
        const = coef * rng.randint(-lim, lim)
    x_parts = _split_sum(rng, coef, 1 + a + b, max_coef)
    c_parts = _split_sum(rng, const, 1 + c + d, max_const)
    if x_parts is None or c_parts is None:
        return None
    # x_L - x_R = coef; c_R - c_L = const (the right side subtracts).
    x_left = x_parts[: 1 + a]
    x_right = [-v for v in x_parts[1 + a :]]
    c_right = c_parts[: 1 + c]
    c_left = [-v for v in c_parts[1 + c :]]
    steps: List[str] = []
    acc = Fraction(x_left[0])
    for t in x_left[1:]:
        op = "+" if t > 0 else "-"
        line, acc = _step(acc, op, abs(t), "combine the x terms")
        steps.append(line)
    for t in x_right:
        term = _render_terms([(abs(t), True)], var)
        if t > 0:
            line, acc = _step(acc, "-", t, f"subtract {term} from both sides")
        else:
            line, acc = _step(acc, "+", -t, f"add {term} to both sides")
        steps.append(line)
    assert acc == coef
    acc = Fraction(c_right[0])
    for t in c_right[1:]:
        op = "+" if t > 0 else "-"
        line, acc = _step(acc, op, abs(t), "combine the constants")
        steps.append(line)
    for t in c_left:
        if t > 0:
            line, acc = _step(acc, "-", t, f"subtract {t} from both sides")
        else:
            line, acc = _step(acc, "+", -t, f"add {-t} to both sides")
        steps.append(line)
    assert acc == const
    if divide:
        line, acc = _step(acc, "/", coef, f"divide by {coef}")
        steps.append(line)
    answer = Fraction(const, coef)
    if len(steps) != n_steps or acc != answer:
        return None
    left = [(v, True) for v in x_left] + [(v, False) for v in c_left]
    right = [(v, True) for v in x_right] + [(v, False) for v in c_right]
    rng.shuffle(left)
    rng.shuffle(right)
    eq = f"{_render_terms(left, var)} = {_render_terms(right, var)}"
    template = rng.choice(
        (
            "Solve {eq} for {v}.",
            "Solve for {v}: {eq}.",
            "Find {v} if {eq}.",
            "What value of {v} satisfies {eq}?",
            "Solve {eq}.",
            "Determine {v} given that {eq}.",
        )
    )
    return lsform.Example(
        task="math",
        problem=template.format(eq=eq, v=var),
        steps=steps,
        answer=fmt_number(answer),
        meta={"kind": "linear", "equation": eq, "variable": var},
    )


# #############################################################################
# Function evaluation and composition
# #############################################################################


def _render_poly(terms: List[Tuple[int, int]], var: str) -> str:
    """
    Render polynomial terms (coefficient, power), highest power first.
    """
    out = []
    for i, (c, k) in enumerate(terms):
        mag = abs(c)
        if k == 0:
            body = str(mag)
        else:
            pw = var if k == 1 else f"{var}^{k}"
            body = pw if mag == 1 else f"{mag}*{pw}"
        if i == 0:
            out.append(f"-{body}" if c < 0 else body)
        else:
            out.append(f"{'-' if c < 0 else '+'} {body}")
    return " ".join(out)


def _random_poly(rng: random.Random, max_coef: int) -> List[Tuple[int, int]]:
    """
    Sample a polynomial of degree 1-3 with distinct powers.
    """
    deg = rng.choice((1, 1, 2, 2, 3))
    powers = [deg] + [k for k in range(deg - 1, -1, -1) if rng.random() < 0.6]
    return [(_nonzero(rng, 1, max_coef), k) for k in powers]


def _eval_poly_steps(
    terms: List[Tuple[int, int]], v: Fraction, max_value: int
) -> Optional[Tuple[List[str], Fraction]]:
    """
    Evaluate a polynomial at `v` one operation per step.

    :return: (steps, value) or None if an intermediate value exceeds
        `max_value` in magnitude
    """
    steps: List[str] = []
    powers: Dict[int, Fraction] = {1: Fraction(v)}

    def power(k: int) -> Fraction:
        if k not in powers:
            prev = power(k - 1)
            line, powers[k] = _step(prev, "*", v)
            steps.append(line)
        return powers[k]

    acc: Optional[Fraction] = None
    for i, (c, k) in enumerate(terms):
        # The first term carries its sign; later terms are added or
        # subtracted by magnitude.
        coef = c if i == 0 else abs(c)
        if k == 0:
            tv = Fraction(coef)
        else:
            p = power(k)
            if coef == 1:
                tv = p
            else:
                line, tv = _step(coef, "*", p)
                steps.append(line)
        if i == 0:
            acc = tv
        else:
            line, acc = _step(acc, "+" if c > 0 else "-", tv)
            steps.append(line)
        if any(abs(x) > max_value for x in list(powers.values()) + [acc, tv]):
            return None
    return steps, acc


def _gen_poly_exact(
    rng: random.Random,
    n_steps: int,
    max_coef: int,
    arg_lim: int,
    max_value: int,
    arg: Optional[Fraction] = None,
) -> Optional[Tuple[List[Tuple[int, int]], Fraction, List[str], Fraction]]:
    """
    Sample (poly, argument) whose evaluation takes exactly `n_steps` steps.
    """
    for _ in range(200):
        terms = _random_poly(rng, max_coef)
        v = Fraction(arg) if arg is not None else Fraction(rng.randint(-arg_lim, arg_lim))
        res = _eval_poly_steps(terms, v, max_value)
        if res is not None and len(res[0]) == n_steps:
            return terms, v, res[0], res[1]
    return None


def _gen_eval(
    rng: random.Random, n_steps: int, max_digits: int, max_value: int
) -> Optional[lsform.Example]:
    max_coef = min(9, 10**max_digits - 1)
    arg_lim = min(10, 10**max_digits - 1)
    var = rng.choice(("x", "x", "t", "n", "y"))
    name = rng.choice(_FUNC_NAMES)
    res = _gen_poly_exact(rng, n_steps, max_coef, arg_lim, max_value)
    if res is None:
        return None
    terms, v, steps, value = res
    expr = _render_poly(terms, var)
    template = rng.choice(
        (
            "Let {f}({x}) = {e}. What is {f}({v})?",
            "Let {f}({x}) = {e}. Calculate {f}({v}).",
            "Suppose {f}({x}) = {e}. Find {f}({v}).",
            "If {f}({x}) = {e}, what is {f}({v})?",
            "Given {f}({x}) = {e}, evaluate {f}({v}).",
        )
    )
    problem = template.format(f=name, x=var, e=expr, v=fmt_number(v))
    return lsform.Example(
        task="math",
        problem=problem,
        steps=steps,
        answer=fmt_number(value),
        meta={
            "kind": "eval",
            "functions": {name: expr},
            "variable": var,
            "call": [name],
            "arg": fmt_number(v),
        },
    )


def _gen_compose(
    rng: random.Random, n_steps: int, max_digits: int, max_value: int
) -> Optional[lsform.Example]:
    if n_steps < 2:
        return None
    max_coef = min(9, 10**max_digits - 1)
    arg_lim = min(10, 10**max_digits - 1)
    var = rng.choice(("x", "x", "t", "n", "y"))
    outer, inner = rng.sample(_FUNC_NAMES, 2)
    n_inner = rng.randint(1, n_steps - 1)
    res_g = _gen_poly_exact(rng, n_inner, max_coef, arg_lim, max_value)
    if res_g is None:
        return None
    g_terms, v, g_steps, w = res_g
    res_f = _gen_poly_exact(rng, n_steps - n_inner, max_coef, arg_lim, max_value, arg=w)
    if res_f is None:
        return None
    f_terms, _, f_steps, value = res_f
    fe, ge = _render_poly(f_terms, var), _render_poly(g_terms, var)
    template = rng.choice(
        (
            "Let {f}({x}) = {fe} and {g}({x}) = {ge}. What is {f}({g}({v}))?",
            "Let {g}({x}) = {ge} and {f}({x}) = {fe}. Calculate {f}({g}({v})).",
            "Suppose {f}({x}) = {fe} and {g}({x}) = {ge}. Find {f}({g}({v})).",
            "If {g}({x}) = {ge} and {f}({x}) = {fe}, what is {f}({g}({v}))?",
        )
    )
    problem = template.format(f=outer, g=inner, x=var, fe=fe, ge=ge, v=fmt_number(v))
    return lsform.Example(
        task="math",
        problem=problem,
        steps=g_steps + f_steps,
        answer=fmt_number(value),
        meta={
            "kind": "compose",
            "functions": {outer: fe, inner: ge},
            "variable": var,
            "call": [outer, inner],
            "arg": fmt_number(v),
        },
    )


# #############################################################################
# Comparisons
# #############################################################################


def _rand_value(
    rng: random.Random, max_digits: int, frac: bool, allow_negative: bool, int_hi: int
) -> Fraction:
    """
    Sample a fraction p/q (q <= 20, |p/q| <= 3) or an integer in [0, int_hi],
    negated with probability 0.3 if allowed.
    """
    hi = 10**max_digits - 1
    if frac:
        den = rng.randint(2, min(20, max(2, hi)))
        num = rng.randint(1, min(hi, 3 * den))
        v = Fraction(num, den)
    else:
        v = Fraction(rng.randint(0, int_hi))
    if allow_negative and rng.random() < 0.3:
        v = -v
    return v


def _compare_steps(b: Fraction, c: Fraction) -> Tuple[List[str], Fraction]:
    """
    Steps whose last value has the sign of b - c (denominators positive).
    """
    steps = []
    if b.denominator == 1 and c.denominator == 1:
        line, d = _step(b, "-", c, "compare")
        return [line], d
    if c.denominator == 1:
        line, cq = _step(c, "*", b.denominator, "cross multiply")
        steps.append(line)
        line, d = _step(b.numerator, "-", cq, "compare")
    elif b.denominator == 1:
        line, bq = _step(b, "*", c.denominator, "cross multiply")
        steps.append(line)
        line, d = _step(bq, "-", c.numerator, "compare")
    else:
        line, x = _step(b.numerator, "*", c.denominator, "cross multiply")
        steps.append(line)
        line, y = _step(c.numerator, "*", b.denominator, "cross multiply")
        steps.append(line)
        line, d = _step(x, "-", y, "compare")
    steps.append(line)
    return steps, d


def _gen_compare(
    rng: random.Random, n_steps: int, max_digits: int, allow_negative: bool
) -> Optional[lsform.Example]:
    k = rng.randint(2, min(n_steps + 1, 6))
    frac_p = rng.random()
    is_frac = [rng.random() < frac_p for _ in range(k)]
    # Next to fractions (|v| <= 3), integers stay small so the comparison
    # is not trivial.
    int_hi = min(5, 10**max_digits - 1) if any(is_frac) else 10**max_digits - 1
    vals = [_rand_value(rng, max_digits, f, allow_negative, int_hi) for f in is_frac]
    if len(set(vals)) != k:
        return None
    want_max = rng.random() < 0.5
    best = vals[0]
    steps: List[str] = []
    for c in vals[1:]:
        s, d = _compare_steps(best, c)
        steps.extend(s)
        if (d < 0) == want_max:
            best = c
    if len(steps) != n_steps:
        return None
    strs = [fmt_number(v) for v in vals]
    big, small = rng.choice(
        (("larger", "smaller"), ("greater", "smaller"), ("bigger", "smaller"))
    )
    adj = big if want_max else small
    sup = {"larger": "largest", "greater": "greatest", "bigger": "biggest"}
    sup["smaller"] = rng.choice(("smallest", "least"))
    if k == 2:
        template = rng.choice(
            (
                "Which is {adj}: {a} or {b}?",
                "Which is {adj}, {a} or {b}?",
                "What is the {adj} of {a} and {b}?",
                "Which number is {adj}: {a} or {b}?",
            )
        )
        problem = template.format(adj=adj, a=strs[0], b=strs[1])
    else:
        lst = rng.choice((", ".join(strs), ", ".join(strs[:-1]) + " and " + strs[-1]))
        template = rng.choice(
            (
                "Which is the {sup} value among {lst}?",
                "What is the {sup} number in {lst}?",
                "Find the {sup} of {lst}.",
                "Which is {sup}: {lst}?",
            )
        )
        problem = template.format(sup=sup[adj], lst=lst)
    return lsform.Example(
        task="math",
        problem=problem,
        steps=steps,
        answer=fmt_number(best),
        meta={"kind": "compare", "values": strs, "mode": "max" if want_max else "min"},
    )


# #############################################################################
# Public API
# #############################################################################


def generate_example(
    rng: random.Random,
    n_steps: int,
    *,
    kind: Optional[str] = None,
    max_digits: int = 3,
    max_value: int = 100_000,
    frac_prob: float = 0.3,
    allow_negative: bool = True,
) -> lsform.Example:
    """
    Generate one M2 example with exactly `n_steps` solution steps.

    :param rng: random generator (controls reproducibility)
    :param n_steps: number of solution steps (>= 1; `compose` needs >= 2)
    :param kind: one of `KINDS`, or None to sample uniformly among the kinds
        that support `n_steps`
    :param max_digits: maximum digits of given constants and integers
    :param max_value: bound on the magnitude of function-evaluation values
    :param frac_prob: probability of a fractional linear-equation solution
    :param allow_negative: allow negative values in comparisons
    :return: math example
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if kind is not None and kind not in KINDS:
        raise ValueError(f"Unknown kind '{kind}'")
    if kind == "compose" and n_steps < 2:
        raise ValueError("compose needs n_steps >= 2")
    if max_digits < 1:
        raise ValueError("max_digits must be >= 1")
    kinds = [kind] if kind else [k for k in KINDS if k != "compose" or n_steps >= 2]
    k = rng.choice(kinds)
    for _ in range(_MAX_ATTEMPTS):
        if k == "linear":
            ex = _gen_linear(rng, n_steps, max_digits, frac_prob)
        elif k == "eval":
            ex = _gen_eval(rng, n_steps, max_digits, max_value)
        elif k == "compose":
            ex = _gen_compose(rng, n_steps, max_digits, max_value)
        else:
            ex = _gen_compare(rng, n_steps, max_digits, allow_negative)
        if ex is not None:
            ex.meta.update(
                {
                    "level": "M2",
                    "n_steps": n_steps,
                    "max_digits": max_digits,
                    "answer_is_fraction": "/" in ex.answer,
                }
            )
            return ex
    raise RuntimeError(f"Could not generate kind={k} with n_steps={n_steps}")


def generate_dataset(
    n: int,
    seed: int,
    *,
    n_steps_range: Tuple[int, int] = (1, 4),
    kinds: Sequence[str] = KINDS,
    max_digits: int = 3,
    max_value: int = 100_000,
    frac_prob: float = 0.3,
    allow_negative: bool = True,
    exclude: Optional[set] = None,
) -> Iterator[lsform.Example]:
    """
    Yield `n` unique M2 examples with `n_steps` uniform in `n_steps_range`
    and the kind uniform among `kinds` (that support the sampled n_steps).

    :param exclude: problem strings to skip (e.g. a held-out test set)
    """
    for k in kinds:
        if k not in KINDS:
            raise ValueError(f"Unknown kind '{k}'")
    rng = random.Random(seed)
    seen = set(exclude or ())
    made = 0
    attempts = 0
    while made < n:
        attempts += 1
        if attempts > 50 * n + 1000:
            raise RuntimeError(f"Only {made}/{n} unique examples; widen the ranges")
        n_steps = rng.randint(*n_steps_range)
        ok = [k for k in kinds if k != "compose" or n_steps >= 2]
        if not ok:
            continue
        ex = generate_example(
            rng,
            n_steps,
            kind=rng.choice(ok),
            max_digits=max_digits,
            max_value=max_value,
            frac_prob=frac_prob,
            allow_negative=allow_negative,
        )
        if ex.problem in seen:
            continue
        seen.add(ex.problem)
        made += 1
        yield ex


def example_to_record(ex: lsform.Example) -> dict:
    """
    Serialize an example to a JSONL record (rendered text + fields).
    """
    return {
        "format": lsform.FORMAT_VERSION,
        "task": ex.task,
        "prompt": lsform.render_prompt(ex),
        "solution": lsform.render_solution(ex),
        "answer": ex.answer,
        **ex.meta,
    }


problem_from_record = lsmam1.problem_from_record


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-steps", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--kinds", nargs="*", default=list(KINDS), choices=KINDS)
    ap.add_argument("--max-digits", type=int, default=3)
    ap.add_argument("--max-value", type=int, default=100_000)
    ap.add_argument("--frac-prob", type=float, default=0.3)
    ap.add_argument("--no-negative", action="store_true")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="JSONL files whose problems must not be generated (test sets)",
    )
    a = ap.parse_args()
    exclude = set()
    for path in a.exclude:
        with open(path) as f:
            exclude.update(problem_from_record(json.loads(line)) for line in f)
    with open(a.out, "w") as f:
        for ex in generate_dataset(
            a.n,
            a.seed,
            n_steps_range=(a.min_steps, a.max_steps),
            kinds=a.kinds,
            max_digits=a.max_digits,
            max_value=a.max_value,
            frac_prob=a.frac_prob,
            allow_negative=not a.no_negative,
            exclude=exclude,
        ):
            f.write(json.dumps(example_to_record(ex)) + "\n")


if __name__ == "__main__":
    _main()
