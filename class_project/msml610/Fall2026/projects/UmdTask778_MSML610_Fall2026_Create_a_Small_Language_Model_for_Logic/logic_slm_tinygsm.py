"""
Convert TinyGSM (word problems with Python solutions) to the v1 worked-step
math format (see `references/format_v1.md`, section 2).

Each TinyGSM row has a `question` and a `code` string holding a function
like:

    def simple_math_problem() -> int:
        \"\"\"<question>\"\"\"
        cookies = 12
        eaten = 4
        left = cookies - eaten
        result = left
        return result

SECURITY: the code strings are untrusted downloaded data. They are never
executed (no exec / eval / compile-to-run). They are parsed with `ast.parse`
and evaluated by a small interpreter over a whitelist, with exact
`fractions.Fraction` arithmetic:

- numeric literals (int / float, not bool / complex / str), names bound
  earlier in the function, unary `-` / `+`, binary `+ - * / // **`;
- calls `abs`, `min`, `max`, `round`, `int`, `sum` (of a literal list or
  tuple) with positional arguments only.

Anything else (loops, ifs, imports, attribute access, strings, other calls,
comprehensions, `%`, function arguments, ...) drops the example, as do
division by zero, values with |value| > 1e9 or denominator > 1e6, and
exponents outside 2..6.

Steps: every binary operation becomes one step `<description>: a op b = c`,
nested expressions are decomposed bottom-up. The description is the
assigned variable name in words (`total_cost` -> `total cost`), also for
the intermediate steps of a nested expression. Constant assignments and
plain copies produce no step. Other operations are mapped as follows:

- `a // b`: written as `a / b = c` when the division is exact, otherwise
  the example is dropped;
- `x ** n` (n in 2..6): n - 1 multiplication steps;
- `sum([a, b, c])`: chained `+` steps;
- `int(x)` / `round(x)`: identity (no step) when x is an integer, otherwise
  the example is dropped (lossy rounding is not expressible);
- `abs` / `min` / `max`: no step is written; later steps use the resulting
  value directly (each emitted step is still checkable on its own);
- unary minus: no step, the negated value is used directly.

Only the steps the returned value depends on are kept (unused "distractor"
computations are pruned). The example is kept only if the answer is an
integer, there is at least one step, and the last step computes the answer.
Every emitted step is verified with `logic_slm_scorers.check_math_step`.

Usage:

    python -I logic_slm_tinygsm.py --out_dir data/processed/real/tinygsm \\
        --workers 17 data/raw/tinygsm/data/train-*.parquet

Import as:

import logic_slm_tinygsm as lsmtgsm
"""

import argparse
import ast
import collections
import fractions
import gzip
import json
import multiprocessing
import os
import re
import sys
import time
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import logic_slm_format as lsform
import logic_slm_scorers as lssco

Fraction = fractions.Fraction

GEN_NAME = "tinygsm"
# Limits on the untrusted input and on the values it computes.
MAX_CODE_CHARS = 4000
MAX_ABS_VALUE = 10**9
MAX_DENOMINATOR = 10**6
MAX_POW_EXPONENT = 6
MAX_EXPR_DEPTH = 30
MAX_STEPS = 40
_ALLOWED_CALLS = ("abs", "min", "max", "round", "int", "sum")
_OP_SYMBOL = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


# #############################################################################
# Interpreter
# #############################################################################


class Rejected(Exception):
    """
    Raised when a row cannot be converted; `reason` is a short drop reason.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def name_to_words(name: str) -> str:
    """
    Turn a variable name into a step description.

    E.g. `total_cost` -> `total cost`, `numApples` -> `num apples`.
    """
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ")
    words = " ".join(words.split()).lower()
    return words or "value"


def format_number(v: Fraction) -> str:
    """
    Render an exact value as an integer or a reduced fraction (`-7/2`).
    """
    if v.denominator == 1:
        return str(v.numerator)
    return f"{v.numerator}/{v.denominator}"


def _check_value(v: Fraction) -> Fraction:
    if abs(v) > MAX_ABS_VALUE:
        raise Rejected("too_large")
    if v.denominator > MAX_DENOMINATOR:
        raise Rejected("bad_fraction")
    return v


# A value with the set of step indices it depends on.
_Val = Tuple[Fraction, FrozenSet[int]]


class _Interpreter:
    """
    Evaluate the whitelisted subset of one function body, recording steps.
    """

    def __init__(self) -> None:
        self.env: Dict[str, _Val] = {}
        # (rendered step, value) in computation order.
        self.steps: List[Tuple[str, Fraction]] = []

    def _emit(self, desc: str, a: _Val, op: str, b: _Val, value: Fraction) -> _Val:
        if len(self.steps) >= MAX_STEPS:
            raise Rejected("too_many_steps")
        step = f"{desc}: {format_number(a[0])} {op} {format_number(b[0])} = {format_number(value)}"
        idx = len(self.steps)
        self.steps.append((step, value))
        return value, a[1] | b[1] | {idx}

    def _binop(self, desc: str, a: _Val, op: str, b: _Val) -> _Val:
        if op == "+":
            v = a[0] + b[0]
        elif op == "-":
            v = a[0] - b[0]
        elif op == "*":
            v = a[0] * b[0]
        else:
            if b[0] == 0:
                raise Rejected("div_by_zero")
            v = a[0] / b[0]
        return self._emit(desc, a, op, b, _check_value(v))

    def eval(self, node: ast.AST, desc: str, depth: int = 0) -> _Val:
        """
        Evaluate an expression node, emitting one step per binary operation.

        :param node: expression AST node
        :param desc: description used for the steps
        :param depth: current nesting depth
        :return: value and the steps it depends on
        """
        if depth > MAX_EXPR_DEPTH:
            raise Rejected("too_deep")
        if isinstance(node, ast.Constant):
            c = node.value
            if isinstance(c, bool) or not isinstance(c, (int, float)):
                raise Rejected("expr:Constant")
            try:
                # repr() of a float is its shortest exact decimal form, so
                # 0.1 becomes 1/10 rather than the binary approximation.
                v = Fraction(c) if isinstance(c, int) else Fraction(repr(c))
            except (ValueError, OverflowError):
                raise Rejected("bad_number")
            return _check_value(v), frozenset()
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise Rejected("unbound_name")
            return self.env[node.id]
        if isinstance(node, ast.UnaryOp):
            v, deps = self.eval(node.operand, desc, depth + 1)
            if isinstance(node.op, ast.USub):
                return -v, deps
            if isinstance(node.op, ast.UAdd):
                return v, deps
            raise Rejected(f"unaryop:{type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node, desc, depth)
        if isinstance(node, ast.Call):
            return self._eval_call(node, desc, depth)
        raise Rejected(f"expr:{type(node).__name__}")

    def _eval_binop(self, node: ast.BinOp, desc: str, depth: int) -> _Val:
        # Check the operator before evaluating the operands.
        op_type = type(node.op)
        if op_type not in _OP_SYMBOL and op_type not in (ast.FloorDiv, ast.Pow):
            raise Rejected(f"binop:{op_type.__name__}")
        a = self.eval(node.left, desc, depth + 1)
        b = self.eval(node.right, desc, depth + 1)
        if op_type in _OP_SYMBOL:
            return self._binop(desc, a, _OP_SYMBOL[op_type], b)
        if op_type is ast.FloorDiv:
            if b[0] == 0:
                raise Rejected("div_by_zero")
            if (a[0] / b[0]).denominator != 1:
                raise Rejected("floordiv_inexact")
            return self._binop(desc, a, "/", b)
        # Pow: small positive integer exponent -> repeated multiplication.
        n = b[0]
        if n.denominator != 1 or not 2 <= n <= MAX_POW_EXPONENT:
            raise Rejected("pow")
        acc = a
        for _ in range(int(n) - 1):
            acc = self._binop(desc, acc, "*", a)
        return acc

    def _eval_call(self, node: ast.Call, desc: str, depth: int) -> _Val:
        f = node.func
        if not isinstance(f, ast.Name) or f.id not in _ALLOWED_CALLS:
            raise Rejected("call")
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise Rejected("call_args")
        name, args = f.id, node.args
        if name == "sum":
            if len(args) != 1 or not isinstance(args[0], (ast.List, ast.Tuple)):
                raise Rejected("call_args")
            elts = args[0].elts
            if not elts or any(isinstance(e, ast.Starred) for e in elts):
                raise Rejected("call_args")
            acc = self.eval(elts[0], desc, depth + 1)
            for e in elts[1:]:
                acc = self._binop(desc, acc, "+", self.eval(e, desc, depth + 1))
            return acc
        if name in ("min", "max"):
            if len(args) < 2:
                raise Rejected("call_args")
            vals = [self.eval(a, desc, depth + 1) for a in args]
            v = min(x[0] for x in vals) if name == "min" else max(x[0] for x in vals)
            deps: FrozenSet[int] = frozenset().union(*(x[1] for x in vals))
            return v, deps
        if name == "abs":
            if len(args) != 1:
                raise Rejected("call_args")
            v, deps = self.eval(args[0], desc, depth + 1)
            return abs(v), deps
        # int(x), round(x), round(x, ndigits): identity on integers only.
        if not (len(args) == 1 or (name == "round" and len(args) == 2)):
            raise Rejected("call_args")
        if len(args) == 2:
            nd = self.eval(args[1], desc, depth + 1)[0]
            if nd.denominator != 1 or not 0 <= nd <= 10:
                raise Rejected("call_args")
        v, deps = self.eval(args[0], desc, depth + 1)
        if v.denominator != 1:
            raise Rejected("lossy_round")
        return v, deps

    def assign(self, target: str, value: ast.AST) -> None:
        if target in _ALLOWED_CALLS:
            raise Rejected("shadows_builtin")
        self.env[target] = self.eval(value, name_to_words(target))


def _function_body(code: str) -> List[ast.stmt]:
    """
    Parse the code and return the body of its single argument-free function.
    """
    if len(code) > MAX_CODE_CHARS:
        raise Rejected("too_long")
    try:
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise Rejected("parse_error")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise Rejected("not_single_function")
    fn = tree.body[0]
    a = fn.args
    if (
        a.args or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg
        or fn.decorator_list
    ):
        raise Rejected("has_args")
    body = list(fn.body)
    # Skip the docstring.
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def _solve(code: str) -> Tuple[List[str], Fraction]:
    """
    Interpret the function and return (kept steps, returned value).
    """
    interp = _Interpreter()
    body = _function_body(code)
    ret: Optional[_Val] = None
    for i, st in enumerate(body):
        if isinstance(st, ast.Assign):
            if len(st.targets) != 1 or not isinstance(st.targets[0], ast.Name):
                raise Rejected("assign_target")
            interp.assign(st.targets[0].id, st.value)
        elif isinstance(st, ast.AnnAssign):
            if st.value is None or not isinstance(st.target, ast.Name):
                raise Rejected("assign_target")
            interp.assign(st.target.id, st.value)
        elif isinstance(st, ast.AugAssign):
            if not isinstance(st.target, ast.Name):
                raise Rejected("assign_target")
            expr = ast.BinOp(left=ast.Name(id=st.target.id, ctx=ast.Load()), op=st.op, right=st.value)
            interp.assign(st.target.id, expr)
        elif isinstance(st, ast.Return):
            if st.value is None:
                raise Rejected("no_return")
            if i != len(body) - 1:
                raise Rejected("code_after_return")
            ret = interp.eval(st.value, "result")
        else:
            raise Rejected(f"stmt:{type(st).__name__}")
    if ret is None:
        raise Rejected("no_return")
    value, deps = ret
    steps = [s for j, (s, _) in enumerate(interp.steps) if j in deps]
    if steps and interp.steps[max(deps)][1] != value:
        raise Rejected("answer_not_last_step")
    return steps, value


def convert_row_with_reason(
    question: Optional[str], code: Optional[str]
) -> Tuple[Optional[lsform.Example], Optional[str]]:
    """
    Convert one TinyGSM row, reporting why it was dropped.

    :param question: problem text
    :param code: Python solution (untrusted; never executed)
    :return: (example, None) if converted, else (None, drop reason)
    """
    if not isinstance(question, str) or not isinstance(code, str):
        return None, "missing_field"
    problem = " ".join(question.split())
    if not problem:
        return None, "empty_question"
    try:
        steps, value = _solve(code)
    except Rejected as e:
        return None, e.reason
    except RecursionError:
        return None, "too_deep"
    if value.denominator != 1:
        return None, "non_integer_answer"
    if not steps:
        return None, "no_steps"
    if not all(lssco.check_math_step(s) is True for s in steps):
        # Should not happen: guards against rendering bugs.
        return None, "step_check_failed"
    ex = lsform.Example(
        task="math",
        problem=problem,
        steps=steps,
        answer=format_number(value),
        meta={"level": GEN_NAME, "n_steps": len(steps)},
    )
    return ex, None


def convert_row(question: Optional[str], code: Optional[str]) -> Optional[lsform.Example]:
    """
    Convert one TinyGSM row to an example, or None if it is dropped.

    :param question: problem text
    :param code: Python solution (untrusted; never executed)
    """
    return convert_row_with_reason(question, code)[0]


def example_to_record(ex: lsform.Example) -> Dict[str, Any]:
    """
    Serialize an example to a JSONL record.
    """
    return {
        "gen": GEN_NAME,
        "task": ex.task,
        "format": lsform.FORMAT_VERSION,
        "prompt": lsform.render_prompt(ex),
        "solution": lsform.render_solution(ex),
        "answer": ex.answer,
        "n_steps": len(ex.steps),
    }


# #############################################################################
# Shard conversion
# #############################################################################


def convert_shard(
    in_path: str, out_path: str, *, batch_size: int = 8192, limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convert one parquet shard to a gzip-compressed JSONL shard.

    :param in_path: parquet file with `question` and `code` columns
    :param out_path: output `.jsonl.gz` path (written atomically)
    :param batch_size: parquet rows per batch
    :param limit: convert at most this many rows
    :return: stats (rows in, kept, drop reason counts, seconds)
    """
    import pyarrow.parquet as pq

    t0 = time.time()
    n_in = n_kept = 0
    drops: collections.Counter = collections.Counter()
    tmp = out_path + ".tmp"
    pf = pq.ParquetFile(in_path)
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
        for batch in pf.iter_batches(batch_size=batch_size, columns=["question", "code"]):
            cols = batch.to_pydict()
            for q, c in zip(cols["question"], cols["code"]):
                if limit is not None and n_in >= limit:
                    break
                n_in += 1
                ex, reason = convert_row_with_reason(q, c)
                if ex is None:
                    drops[reason] += 1
                    continue
                n_kept += 1
                f.write(json.dumps(example_to_record(ex), ensure_ascii=False) + "\n")
            if limit is not None and n_in >= limit:
                break
    os.replace(tmp, out_path)
    return {
        "shard": os.path.basename(in_path),
        "out": out_path,
        "rows_in": n_in,
        "kept": n_kept,
        "drops": dict(drops.most_common()),
        "seconds": round(time.time() - t0, 1),
    }


def _out_path(in_path: str, out_dir: str) -> str:
    stem = os.path.basename(in_path)
    if stem.endswith(".parquet"):
        stem = stem[: -len(".parquet")]
    return os.path.join(out_dir, f"{GEN_NAME}_{stem}.jsonl.gz")


def _convert_shard_task(args: Tuple[str, str, int, Optional[int]]) -> Dict[str, Any]:
    in_path, out_path, batch_size, limit = args
    return convert_shard(in_path, out_path, batch_size=batch_size, limit=limit)


def convert_shards(
    in_paths: Sequence[str],
    out_dir: str,
    *,
    workers: int = 1,
    batch_size: int = 8192,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert parquet shards in parallel, one output shard per input shard.

    :return: per-shard stats, in input order
    """
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(p, _out_path(p, out_dir), batch_size, limit) for p in in_paths]
    stats: List[Dict[str, Any]] = []
    with multiprocessing.Pool(max(1, min(workers, len(tasks)))) as pool:
        for st in pool.imap_unordered(_convert_shard_task, tasks):
            _print_stats(st)
            stats.append(st)
    order = {os.path.basename(p): i for i, p in enumerate(in_paths)}
    stats.sort(key=lambda s: order[s["shard"]])
    return stats


def _print_stats(st: Dict[str, Any]) -> None:
    rate = st["kept"] / max(1, st["rows_in"])
    print(
        f"{st['shard']}: rows_in={st['rows_in']} kept={st['kept']} "
        f"({rate:.1%}) seconds={st['seconds']} drops={json.dumps(st['drops'])}",
        flush=True,
    )


def total_stats(stats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sum per-shard stats.
    """
    drops: collections.Counter = collections.Counter()
    for st in stats:
        drops.update(st["drops"])
    rows_in = sum(st["rows_in"] for st in stats)
    kept = sum(st["kept"] for st in stats)
    return {
        "shards": len(stats),
        "rows_in": rows_in,
        "kept": kept,
        "keep_rate": round(kept / max(1, rows_in), 4),
        "drops": dict(drops.most_common()),
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="TinyGSM parquet shards")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None, help="max rows per shard")
    args = ap.parse_args()
    stats = convert_shards(
        args.inputs, args.out_dir, workers=args.workers,
        batch_size=args.batch_size, limit=args.limit,
    )
    tot = total_stats(stats)
    print("TOTAL", json.dumps(tot), flush=True)
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump({"total": tot, "shards": stats}, f, indent=1)


if __name__ == "__main__":
    _main()
