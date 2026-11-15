"""
Unit tests for `logic_slm_math_m2.py`.

Import as:

import test.test_logic_slm_math_m2 as ttlsmm2
"""

import random
import re

import pytest
import sympy
from sympy.parsing.sympy_parser import parse_expr

import logic_slm_format as lsform
import logic_slm_math_m2 as lsmam2
import logic_slm_scorers as lssco

_N = 1000


def _sym(expr: str, var: str) -> sympy.Expr:
    """
    Parse a generator expression (`3*x^2 - 5`) with sympy.
    """
    return parse_expr(expr.replace("^", "**"), local_dict={var: sympy.Symbol(var)})


def _sympy_answer(ex: lsform.Example) -> sympy.Rational:
    """
    Compute the answer independently with sympy from the problem data.
    """
    m = ex.meta
    if m["kind"] == "linear":
        assert m["equation"] in ex.problem
        var = sympy.Symbol(m["variable"])
        lhs, rhs = m["equation"].split(" = ")
        sol = sympy.solve(sympy.Eq(_sym(lhs, m["variable"]), _sym(rhs, m["variable"])), var)
        assert len(sol) == 1, ex.problem
        return sympy.Rational(sol[0])
    if m["kind"] in ("eval", "compose"):
        var = sympy.Symbol(m["variable"])
        val = sympy.Rational(m["arg"])
        for name in reversed(m["call"]):
            assert f"{name}({m['variable']}) = {m['functions'][name]}" in ex.problem
            val = _sym(m["functions"][name], m["variable"]).subs(var, val)
        return sympy.Rational(val)
    assert m["kind"] == "compare"
    for v in m["values"]:
        assert v in ex.problem
    vals = [sympy.Rational(v) for v in m["values"]]
    return max(vals) if m["mode"] == "max" else min(vals)


def _examples(n=_N, seed=0, **kw):
    kw.setdefault("n_steps_range", (1, 8))
    return list(lsmam2.generate_dataset(n, seed, **kw))


def _last_value(step: str) -> str:
    return step.rsplit("=", 1)[1].strip()


class Test_generate:
    def test_answers_match_sympy(self) -> None:
        for ex in _examples():
            assert sympy.Rational(ex.answer) == _sympy_answer(ex), ex.problem

    def test_answer_canonical(self) -> None:
        for ex in _examples(500):
            assert lsform.normalize_answer(ex.answer, "math") == ex.answer

    def test_steps_are_valid(self) -> None:
        for ex in _examples():
            assert len(ex.steps) == ex.meta["n_steps"]
            for s in ex.steps:
                assert lssco.check_math_step(s) is True, (ex.problem, s)
            if ex.meta["kind"] != "compare":
                assert lsform.normalize_answer(_last_value(ex.steps[-1]), "math") == ex.answer

    def test_compare_last_step_sign(self) -> None:
        # For two values the last difference has the sign of first - second.
        for ex in _examples(600, kinds=("compare",)):
            if len(ex.meta["values"]) == 2:
                a, b = (sympy.Rational(v) for v in ex.meta["values"])
                d = sympy.Rational(_last_value(ex.steps[-1]))
                assert sympy.sign(d) == sympy.sign(a - b), ex.problem

    def test_all_kinds_and_fractions(self) -> None:
        exs = _examples()
        assert {e.meta["kind"] for e in exs} == set(lsmam2.KINDS)
        lin = [e for e in exs if e.meta["kind"] == "linear"]
        frac = sum(e.meta["answer_is_fraction"] for e in lin) / len(lin)
        assert 0.1 < frac < 0.6
        assert any(e.answer.startswith("-") for e in exs)

    def test_each_kind_each_step_count(self) -> None:
        rng = random.Random(3)
        for kind in lsmam2.KINDS:
            for n_steps in range(2 if kind == "compose" else 1, 9):
                ex = lsmam2.generate_example(rng, n_steps, kind=kind)
                assert ex.meta["kind"] == kind and len(ex.steps) == n_steps

    def test_max_digits(self) -> None:
        for ex in _examples(500, max_digits=1, kinds=("linear", "compare")):
            nums = re.findall(r"\d+", ex.problem)
            assert all(len(x) <= 1 for x in nums), ex.problem

    def test_round_trip_through_format(self) -> None:
        for ex in _examples(300):
            text = lsform.render_example(ex)
            assert lsform.parse_answer(text) == ex.answer
            assert lsform.parse_steps(text) == ex.steps
            assert lssco.math_steps_valid(text)


class Test_dataset:
    def test_deterministic(self) -> None:
        a = [lsform.render_example(e) for e in _examples(200, seed=7)]
        b = [lsform.render_example(e) for e in _examples(200, seed=7)]
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
            rec = lsmam2.example_to_record(ex)
            assert lsmam2.problem_from_record(rec) == ex.problem
            assert rec["kind"] == ex.meta["kind"]

    def test_n_steps_range(self) -> None:
        exs = _examples(400, n_steps_range=(5, 8))
        assert {e.meta["n_steps"] for e in exs} == {5, 6, 7, 8}
        exs = _examples(400, n_steps_range=(1, 4))
        assert {len(e.steps) for e in exs} == {1, 2, 3, 4}

    def test_invalid_args(self) -> None:
        rng = random.Random(0)
        with pytest.raises(ValueError):
            lsmam2.generate_example(rng, 0)
        with pytest.raises(ValueError):
            lsmam2.generate_example(rng, 1, kind="compose")
        with pytest.raises(ValueError):
            lsmam2.generate_example(rng, 2, kind="nope")
