"""
Unit tests for `logic_slm/math_m1.py`.

Import as:

import test.test_math_m1 as ttmam1
"""

import ast
import fractions
import operator
import random
import re

import pytest

import logic_slm.format as lsform
import logic_slm.math_m1 as lsmam1

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_STEP_RE = re.compile(r"^(\d+) ([-+*/]) (\d+) = (\d+)$")


def _exact_eval(expr: str) -> fractions.Fraction:
    """
    Evaluate an arithmetic expression exactly (independent of the generator).
    """

    def ev(n):
        if isinstance(n, ast.Constant):
            return fractions.Fraction(n.value)
        assert isinstance(n, ast.BinOp), ast.dump(n)
        return _BINOPS[type(n.op)](ev(n.left), ev(n.right))

    return ev(ast.parse(expr, mode="eval").body)


def _examples(n=2000, seed=0, **kw):
    return list(lsmam1.generate_dataset(n, seed, n_ops_range=(1, 8), **kw))


class Test_generate:
    def test_answer_matches_exact_eval(self) -> None:
        for ex in _examples():
            expr = ex.problem[: -len(" = ?")]
            assert _exact_eval(expr) == int(ex.answer), ex.problem

    def test_steps_are_valid(self) -> None:
        for ex in _examples():
            assert len(ex.steps) == ex.meta["n_ops"]
            for s in ex.steps:
                m = _STEP_RE.match(s)
                assert m, s
                a, op, b, c = int(m[1]), m[2], int(m[3]), int(m[4])
                assert _exact_eval(f"{a} {op} {b}") == c, s
            assert ex.steps[-1].endswith(f"= {ex.answer}")

    def test_bounds(self) -> None:
        for ex in _examples(max_digits=2, max_value=500):
            for s in ex.steps:
                vals = [int(v) for v in re.findall(r"\d+", s)]
                assert all(0 <= v <= 500 for v in vals), s
            for leaf in re.findall(r"\d+", ex.problem):
                assert len(leaf) <= 2

    def test_brackets_occur(self) -> None:
        exs = _examples()
        frac = sum(ex.meta["has_brackets"] for ex in exs) / len(exs)
        assert 0.2 < frac < 0.95

    def test_all_ops_occur(self) -> None:
        text = " ".join(ex.problem for ex in _examples())
        for op in lsmam1.OPS:
            assert f" {op} " in text

    def test_round_trip_through_format(self) -> None:
        for ex in _examples(200):
            text = lsform.render_example(ex)
            assert lsform.parse_answer(text) == ex.answer
            assert lsform.parse_steps(text) == ex.steps


class Test_dataset:
    def test_deterministic(self) -> None:
        a = [lsform.render_example(e) for e in _examples(100, seed=7)]
        b = [lsform.render_example(e) for e in _examples(100, seed=7)]
        assert a == b

    def test_unique_and_exclude(self) -> None:
        exs = _examples(500, seed=1)
        probs = [e.problem for e in exs]
        assert len(set(probs)) == len(probs)
        excl = set(probs[:100])
        exs2 = _examples(500, seed=1, exclude=excl)
        assert not excl & {e.problem for e in exs2}

    def test_problem_from_record(self) -> None:
        for ex in _examples(50):
            rec = lsmam1.example_to_record(ex)
            assert lsmam1.problem_from_record(rec) == ex.problem

    def test_n_ops_range(self) -> None:
        exs = list(lsmam1.generate_dataset(300, 0, n_ops_range=(5, 8)))
        assert {e.meta["n_ops"] for e in exs} == {5, 6, 7, 8}

    def test_invalid_args(self) -> None:
        with pytest.raises(ValueError):
            lsmam1.generate_example(random.Random(0), 0)
        with pytest.raises(ValueError):
            lsmam1.generate_example(random.Random(0), 2, max_digits=3, max_value=50)
