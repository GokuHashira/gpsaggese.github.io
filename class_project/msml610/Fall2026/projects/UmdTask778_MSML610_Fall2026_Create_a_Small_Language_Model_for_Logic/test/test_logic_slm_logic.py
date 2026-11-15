"""
Unit tests for `logic_slm_logic.py`.

Labels and steps are re-checked with an independent z3 encoding (and, for
L1, a brute-force truth table) built here from the parsed formulas.

Import as:

import test.test_logic_slm_logic as ttlslo
"""

import collections
import functools
import itertools
import json
import random
from typing import Dict, List

import pytest
import z3

import logic_slm_format as lsform
import logic_slm_logic as lslog

_N = 240


@functools.lru_cache(maxsize=None)
def _dataset(level: str, n: int = _N, seed: int = 0, split: str = "train", depth=(1, 4)):
    return tuple(
        lslog.generate_dataset(n, seed, level=level, template_split=split, depth_range=depth)
    )


def _symbolic(ex: lsform.Example):
    """
    Return (premises text, question text) of the symbolic source.
    """
    if ex.meta["level"] == "L3":
        return ex.meta["sym_premises"], ex.meta["sym_question"]
    return ex.problem, ex.question


# #############################################################################
# Independent checkers
# #############################################################################

_OBJ = z3.DeclareSort("TestObj")


def _z3(f, env: Dict[str, z3.ExprRef]) -> z3.BoolRef:
    op = f[0]
    if op == "v":
        return z3.Bool("t_" + f[1])
    if op == "P":
        term = env[f[2]] if f[2] in env else z3.Const("t_" + f[2], _OBJ)
        return z3.Function("t_" + f[1], _OBJ, z3.BoolSort())(term)
    if op == "~":
        return z3.Not(_z3(f[1], env))
    if op == "forall":
        x = z3.FreshConst(_OBJ)
        return z3.ForAll([x], _z3(f[2], {**env, f[1]: x}))
    a, b = _z3(f[1], env), _z3(f[2], env)
    return {"&": z3.And, "|": z3.Or, "->": z3.Implies}[op](a, b)


def _entails(premises, goal) -> bool:
    s = z3.Solver()
    for f in premises:
        s.add(_z3(f, {}))
    s.add(z3.Not(_z3(goal, {})))
    r = s.check()
    assert r != z3.unknown
    return r == z3.unsat


def _consistent(premises) -> bool:
    s = z3.Solver()
    for f in premises:
        s.add(_z3(f, {}))
    return s.check() == z3.sat


def _truth_table_verdict(premises, q) -> str:
    """
    Brute-force L1 verdict over all assignments.
    """

    def ev(f, m):
        op = f[0]
        if op == "v":
            return m[f[1]]
        if op == "~":
            return not ev(f[1], m)
        a, b = ev(f[1], m), ev(f[2], m)
        return {"&": a and b, "|": a or b, "->": (not a) or b}[op]

    atoms = sorted(lslog._atoms(list(premises) + [q]))
    vals = set()
    for bits in itertools.product((False, True), repeat=len(atoms)):
        m = dict(zip(atoms, bits))
        if all(ev(f, m) for f in premises):
            vals.add(ev(q, m))
    assert vals, "inconsistent premises"
    if vals == {True}:
        return "yes"
    if vals == {False}:
        return "no"
    return "unknown"


# #############################################################################
# Tests
# #############################################################################


@pytest.mark.parametrize("level", lslog.LEVELS)
class Test_dataset_properties:
    def test_balanced(self, level: str) -> None:
        cnt = collections.Counter(e.answer for e in _dataset(level))
        assert set(cnt) == set(lsform.LOGIC_VERDICTS)
        for v in lsform.LOGIC_VERDICTS:
            assert abs(cnt[v] - _N / 3) <= 1, cnt

    def test_labels_z3(self, level: str) -> None:
        for ex in _dataset(level):
            ptext, qtext = _symbolic(ex)
            prem = lslog.parse_premises(ptext)
            q = lslog.parse_formula(qtext)
            assert _consistent(prem), ptext
            yes, no = _entails(prem, q), _entails(prem, lslog.neg(q))
            expected = "yes" if yes else "no" if no else "unknown"
            assert ex.answer == expected, (ptext, qtext)
            if ex.meta["source"] == "L1":
                assert _truth_table_verdict(prem, q) == ex.answer

    def test_round_trip_through_format(self, level: str) -> None:
        for ex in _dataset(level):
            text = lsform.render_example(ex)
            assert lsform.parse_answer(text) == ex.answer
            assert lsform.parse_steps(text) == ex.steps
            assert text.split("\n")[1] == f"Question: {ex.question}?"
            assert ex.problem.endswith(".")

    def test_meta(self, level: str) -> None:
        for ex in _dataset(level):
            m = ex.meta
            assert m["level"] == level and m["verdict"] == ex.answer
            assert m["depth"] == len(ex.steps)
            assert 1 <= m["depth"] <= 4
            assert m["n_premises"] <= max(5, m["depth"] + 2)
            rec = lslog.example_to_record(ex)
            rec = json.loads(json.dumps(rec))
            for k in ("prompt", "solution", "answer", "task", "format", "level", "depth"):
                assert k in rec
            assert lslog.problem_from_record(rec) == lslog.example_key(ex)


@pytest.mark.parametrize("level", ("L1", "L2"))
class Test_symbolic:
    def test_formula_round_trip(self, level: str) -> None:
        for ex in _dataset(level):
            prem = lslog.parse_premises(ex.problem)
            assert ". ".join(map(lslog.render_formula, prem)) + "." == ex.problem
            assert len(prem) == ex.meta["n_premises"]
            for s in ex.steps:
                ante, concl = lslog.parse_step(s)
                assert lslog.render_step((tuple(ante), concl, "")) == s

    def test_steps_valid_and_available(self, level: str) -> None:
        for ex in _dataset(level):
            avail = set(lslog.parse_premises(ex.problem))
            q = lslog.parse_formula(ex.question)
            for s in ex.steps:
                ante, concl = lslog.parse_step(s)
                assert all(f in avail for f in ante), (ex.problem, s)
                assert _entails(ante, concl), s
                assert lslog.check_logic_step(s) is True, s
                avail.add(concl)
            concls = [lslog.parse_step(s)[1] for s in ex.steps]
            if ex.answer == "yes":
                assert concls[-1] == q
            elif ex.answer == "no":
                assert concls[-1] == lslog.neg(q)
            else:
                assert q not in concls and lslog.neg(q) not in concls

    def test_syntax(self, level: str) -> None:
        text = " ".join(lsform.render_example(e) for e in _dataset(level))
        if level == "L1":
            assert "forall" not in text
            assert set(lslog._atoms(lslog.parse_premises(_dataset(level)[0].problem))) <= set(
                lslog.ATOMS + lslog.EXTRA_ATOMS
            )
        else:
            assert "forall x (" in text and " => " in text


class Test_check_logic_step:
    @pytest.mark.parametrize(
        "step,expected",
        [
            ("p, p -> q => q", True),
            ("~q, p -> q => ~p", True),
            ("p | q, ~p => q", True),
            ("p & q => p", True),
            ("p, q => p & q", True),
            ("p & q, p & q -> r => r", True),
            ("q, p -> q => p", False),
            ("p | q => q", False),
            ("forall x (Cat(x) -> Mammal(x)), Cat(tom) => Mammal(tom)", True),
            ("forall x (Cat(x) -> ~Bird(x)), Bird(ann) => ~Cat(ann)", True),
            ("forall x (Cat(x) -> Mammal(x)), Cat(tom) => Mammal(ann)", False),
            ("Tom is a cat, all cats are mammals => Tom is a mammal", None),
            ("p -> q", None),
            ("p, => q", None),
            ("17 + 4 = 21", None),
        ],
    )
    def test_cases(self, step: str, expected) -> None:
        assert lslog.check_logic_step(step) is expected

    def test_english_steps_unparseable(self) -> None:
        for ex in _dataset("L3")[:60]:
            for s in ex.steps:
                assert lslog.check_logic_step(s) is None, s


class Test_L3:
    def test_templates_disjoint(self) -> None:
        train = set().union(*(e.meta["templates"] for e in _dataset("L3")))
        held = set().union(*(e.meta["templates"] for e in _dataset("L3", split="heldout")))
        assert train and held and not train & held
        all_ids = {tid for c in lslog.TEMPLATES.values() for tid, _ in c["train"]}
        assert train <= all_ids

    def test_both_sources(self) -> None:
        srcs = collections.Counter(e.meta["source"] for e in _dataset("L3"))
        assert srcs["L1"] > 50 and srcs["L2"] > 50

    def test_english_text(self) -> None:
        for ex in _dataset("L3"):
            assert not any(t in ex.problem for t in ("->", "forall", "~", "&"))
            assert ex.problem[0].isupper() and ex.question[0].isupper()
            assert ex.problem.count(". ") == ex.meta["n_premises"] - 1


class Test_generation:
    def test_deterministic(self) -> None:
        for level in lslog.LEVELS:
            a = [lsform.render_example(e) for e in lslog.generate_dataset(60, 7, level=level)]
            b = [lsform.render_example(e) for e in lslog.generate_dataset(60, 7, level=level)]
            assert a == b

    def test_unique_and_exclude(self) -> None:
        exs = list(lslog.generate_dataset(300, 1, level="L1", depth_range=(1, 2)))
        keys = [lslog.example_key(e) for e in exs]
        assert len(set(keys)) == len(keys)
        excl = set(keys[:100])
        exs2 = lslog.generate_dataset(300, 1, level="L1", depth_range=(1, 2), exclude=excl)
        assert not excl & {lslog.example_key(e) for e in exs2}

    @pytest.mark.parametrize("level", lslog.LEVELS)
    def test_depth_range(self, level: str) -> None:
        exs = _dataset(level, n=90, seed=3, depth=(5, 6))
        assert {e.meta["depth"] for e in exs} == {5, 6}
        assert all(len(e.steps) == e.meta["depth"] for e in exs)

    def test_unknown_may_be_empty(self) -> None:
        exs = list(lslog.generate_dataset(30, 0, level="L1", depth_range=(0, 1)))
        assert any(e.answer == "unknown" and not e.steps for e in exs)
        assert all(e.steps for e in exs if e.answer != "unknown")

    def test_max_premises(self) -> None:
        exs = list(lslog.generate_dataset(60, 0, level="L2", depth_range=(1, 2), max_premises=4))
        assert all(e.meta["n_premises"] <= 4 for e in exs)

    def test_example_args(self) -> None:
        rng = random.Random(0)
        ex = lslog.generate_example(rng, "L2", verdict="no", depth=3)
        assert ex.answer == "no" and ex.meta["depth"] == 3
        with pytest.raises(ValueError):
            lslog.generate_example(rng, "L4")
        with pytest.raises(ValueError):
            lslog.generate_example(rng, "L1", verdict="yes", depth=0)
        with pytest.raises(ValueError):
            lslog.generate_example(rng, "L3", template_split="test")
