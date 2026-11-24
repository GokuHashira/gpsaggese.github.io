"""
M1 generator: arithmetic expressions with worked steps.

An expression is a random binary tree with `n_ops` operators over
non-negative integer leaves. Values stay non-negative integers: subtraction
swaps its operands when needed, division is only kept when exact, and
products / sums above `max_value` are replaced by a smaller operation.

The solution evaluates the tree bottom-up, left to right, one operation per
step (`a op b = c`), as specified in `references/format_v1.md`. With
`state_steps=True` every step except the last also rewrites the remaining
expression (`15 * 69 = 1035, so 1 * 1035 + 674 - 3 + 6`), so the model never
has to remember which operations are still pending.

Import as:

import logic_slm_math_m1 as lsmam1
"""

import argparse
import copy
import dataclasses
import json
import random
from typing import Iterator, List, Optional, Tuple, Union

import logic_slm_format as lsform

OPS = ("+", "-", "*", "/")
_PREC = {"+": 1, "-": 1, "*": 2, "/": 2}


@dataclasses.dataclass
class _Node:
    """
    Expression tree node: a leaf holds `value`; an inner node holds `op`.
    """

    value: int = 0
    op: Optional[str] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.op is None


def _rand_number(rng: random.Random, max_digits: int) -> int:
    """
    Sample a number whose digit count is uniform in [1, max_digits].
    """
    d = rng.randint(1, max_digits)
    lo = 0 if d == 1 else 10 ** (d - 1)
    return rng.randint(lo, 10**d - 1)


def _build_tree(rng: random.Random, n_ops: int, max_digits: int, ops: Tuple[str, ...]) -> _Node:
    """
    Build a random tree shape with `n_ops` inner nodes and random leaves.
    """
    if n_ops == 0:
        return _Node(value=_rand_number(rng, max_digits))
    n_left = rng.randint(0, n_ops - 1)
    return _Node(
        op=rng.choice(ops),
        left=_build_tree(rng, n_left, max_digits, ops),
        right=_build_tree(rng, n_ops - 1 - n_left, max_digits, ops),
    )


def _fix_and_eval(node: _Node, rng: random.Random, max_value: int, max_leaf: int) -> int:
    """
    Evaluate the tree bottom-up, repairing operations so every intermediate
    value is a non-negative integer <= `max_value`. Leaf divisors chosen
    here stay <= `max_leaf`.
    """
    if node.is_leaf:
        return node.value
    lv = _fix_and_eval(node.left, rng, max_value, max_leaf)
    rv = _fix_and_eval(node.right, rng, max_value, max_leaf)
    if node.op == "/":
        if node.right.is_leaf and lv > 0:
            # Make the leaf divisor a random divisor of the left value.
            divs = [d for d in range(1, min(lv, max_leaf) + 1) if lv % d == 0]
            node.right.value = rv = rng.choice(divs)
        if rv == 0 or lv % rv != 0:
            node.op = rng.choice(("+", "-", "*"))
    if node.op == "*" and lv * rv > max_value:
        node.op = "+"
    if node.op == "+" and lv + rv > max_value:
        node.op = "-"
    if node.op == "-" and lv < rv:
        node.left, node.right = node.right, node.left
        lv, rv = rv, lv
    if node.op == "+":
        node.value = lv + rv
    elif node.op == "-":
        node.value = lv - rv
    elif node.op == "*":
        node.value = lv * rv
    else:
        node.value = lv // rv
    return node.value


def _render(node: _Node) -> str:
    """
    Render with the minimal brackets needed to preserve the tree.
    """
    if node.is_leaf:
        return str(node.value)
    p = _PREC[node.op]
    left = _render(node.left)
    if not node.left.is_leaf and _PREC[node.left.op] < p:
        left = f"({left})"
    right = _render(node.right)
    if not node.right.is_leaf:
        rp = _PREC[node.right.op]
        if rp < p or (rp == p and node.op in ("-", "/")):
            right = f"({right})"
    return f"{left} {node.op} {right}"


def _steps(node: _Node, out: List[str]) -> None:
    """
    Append one step per operation in post-order (left subtree first).
    """
    if node.is_leaf:
        return
    _steps(node.left, out)
    _steps(node.right, out)
    out.append(f"{node.left.value} {node.op} {node.right.value} = {node.value}")


def _parse(text: str) -> _Node:
    """
    Parse a rendered expression into the conventional tree (brackets, then
    `*` `/`, then `+` `-`, each left to right), with node values.

    :raises ValueError: if a value would be negative or a division inexact
    """
    toks = text.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def apply(op: str, left: _Node, right: _Node) -> _Node:
        a, b = left.value, right.value
        if op == "-" and a < b or op == "/" and (b == 0 or a % b):
            raise ValueError("not a valid left-to-right evaluation")
        value = {"+": a + b, "-": a - b, "*": a * b, "/": a // max(b, 1)}[op]
        return _Node(value=value, op=op, left=left, right=right)

    def atom() -> _Node:
        nonlocal pos
        tok = toks[pos]
        pos += 1
        if tok == "(":
            node = expr()
            pos += 1  # ")"
            return node
        return _Node(value=int(tok))

    def term() -> _Node:
        nonlocal pos
        node = atom()
        while pos < len(toks) and toks[pos] in ("*", "/"):
            op = toks[pos]
            pos += 1
            node = apply(op, node, atom())
        return node

    def expr() -> _Node:
        nonlocal pos
        node = term()
        while pos < len(toks) and toks[pos] in ("+", "-"):
            op = toks[pos]
            pos += 1
            node = apply(op, node, term())
        return node

    return expr()


def _state_steps(root: _Node) -> List[str]:
    """
    Steps in the same order as `_steps`, each followed by the expression left
    after replacing the evaluated operation with its value.
    """
    order: List[_Node] = []

    def visit(node: _Node) -> None:
        if node.is_leaf:
            return
        visit(node.left)
        visit(node.right)
        order.append(node)

    visit(root)
    out = []
    for i, node in enumerate(order):
        step = f"{node.left.value} {node.op} {node.right.value} = {node.value}"
        # Collapse the node into a leaf (the tree is a private copy).
        node.op, node.left, node.right = None, None, None
        out.append(step if i == len(order) - 1 else f"{step}, so {_render(root)}")
    return out


def _depth(node: _Node) -> int:
    if node.is_leaf:
        return 0
    return 1 + max(_depth(node.left), _depth(node.right))


def generate_example(
    rng: random.Random,
    n_ops: int,
    *,
    max_digits: int = 3,
    ops: Tuple[str, ...] = OPS,
    max_value: int = 10_000,
    state_steps: bool = False,
) -> lsform.Example:
    """
    Generate one M1 example.

    :param rng: random generator (controls reproducibility)
    :param n_ops: number of operations = number of solution steps (>= 1)
    :param max_digits: maximum digits of each leaf number
    :param ops: operators to sample from
    :param max_value: bound on every intermediate value
    :param state_steps: also rewrite the remaining expression after each step
    :return: math example with one step per operation
    """
    if n_ops < 1:
        raise ValueError("n_ops must be >= 1")
    if max_value < 10**max_digits - 1:
        raise ValueError("max_value must be at least the largest leaf")
    root = _build_tree(rng, n_ops, max_digits, tuple(ops))
    answer = _fix_and_eval(root, rng, max_value, 10**max_digits - 1)
    expr = _render(root)
    depth = _depth(root)
    steps: List[str] = []
    if state_steps:
        # Evaluate in the conventional order of the rendered expression when
        # it stays valid (non-negative, exact division); else in tree order.
        try:
            tree = _parse(expr)
        except ValueError:
            tree = copy.deepcopy(root)
        if tree.value != answer:
            tree = copy.deepcopy(root)
        steps = _state_steps(tree)
    else:
        _steps(root, steps)
    return lsform.Example(
        task="math",
        problem=f"{expr} = ?",
        steps=steps,
        answer=str(answer),
        meta={
            "level": "M1",
            "n_ops": n_ops,
            "depth": depth,
            "max_digits": max_digits,
            "has_brackets": "(" in expr,
        },
    )


def generate_dataset(
    n: int,
    seed: int,
    *,
    n_ops_range: Tuple[int, int] = (1, 4),
    max_digits: int = 3,
    ops: Tuple[str, ...] = OPS,
    max_value: int = 10_000,
    exclude: Optional[set] = None,
    state_steps: bool = False,
) -> Iterator[lsform.Example]:
    """
    Yield `n` unique M1 examples with `n_ops` uniform in `n_ops_range`.

    :param exclude: problem strings to skip (e.g. a held-out test set)
    """
    rng = random.Random(seed)
    seen = set(exclude or ())
    made = 0
    attempts = 0
    while made < n:
        attempts += 1
        if attempts > 50 * n + 1000:
            raise RuntimeError(f"Only {made}/{n} unique examples; widen the ranges")
        ex = generate_example(
            rng,
            rng.randint(*n_ops_range),
            max_digits=max_digits,
            ops=ops,
            max_value=max_value,
            state_steps=state_steps,
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


def problem_from_record(record: dict) -> str:
    """
    Recover the problem string (as in `Example.problem`) from a JSONL record.
    """
    first = record["prompt"].split("\n", 1)[0]
    if not first.startswith("Problem: "):
        raise ValueError(f"Not a math record: {first!r}")
    return first[len("Problem: ") :]


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-ops", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=4)
    ap.add_argument("--max-digits", type=int, default=3)
    ap.add_argument("--max-value", type=int, default=10_000)
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
            n_ops_range=(a.min_ops, a.max_ops),
            max_digits=a.max_digits,
            max_value=a.max_value,
            exclude=exclude,
        ):
            f.write(json.dumps(example_to_record(ex)) + "\n")


if __name__ == "__main__":
    _main()
