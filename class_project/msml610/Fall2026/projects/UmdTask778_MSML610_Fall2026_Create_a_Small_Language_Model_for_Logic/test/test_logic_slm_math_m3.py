"""
Unit tests for `logic_slm_math_m3.py`.

Import as:

import test.test_logic_slm_math_m3 as ttlsmm3
"""

import random
import re

import pytest

import logic_slm_format as lsform
import logic_slm_math_m3 as lsmam3
import logic_slm_scorers as lssco

_N = 1000


def _examples(n=_N, seed=0, **kw):
    kw.setdefault("n_steps_range", (1, 8))
    kw.setdefault("n_distractors_range", (0, 3))
    return list(lsmam3.generate_dataset(n, seed, **kw))


def _value(step: str) -> int:
    return int(step.rsplit("=", 1)[1])


class Test_generate:
    def test_steps_are_valid(self) -> None:
        for ex in _examples():
            assert len(ex.steps) == ex.meta["n_steps"]
            for s in ex.steps:
                assert lssco.check_math_step(s) is True, (ex.problem, s)
                desc = s.split(":")[0]
                assert len(s.split(":")) == 2 and desc and not desc[0].isdigit(), s
            assert _value(ex.steps[-1]) == int(ex.answer)

    def test_answer_from_graph(self) -> None:
        for ex in _examples():
            g = ex.meta["graph"]
            assert len(g) == len(ex.steps)
            assert lsmam3.evaluate_graph(g, ex.meta["given"]) == int(ex.answer)
            # Steps follow the graph nodes in order, and each node only uses
            # earlier nodes (topological order).
            for i, (node, s) in enumerate(zip(g, ex.steps)):
                assert _value(s) == node["value"]
                for a in node["args"]:
                    if isinstance(a, str) and a.startswith("s"):
                        assert int(a[1:]) < i

    def test_values_non_negative_and_given_in_text(self) -> None:
        for ex in _examples():
            assert int(ex.answer) >= 0 and lsform.normalize_answer(ex.answer, "math") == ex.answer
            for node in ex.meta["graph"]:
                assert node["value"] >= 0
            nums = set(re.findall(r"\d+", ex.problem))
            for v in ex.meta["given"]:
                assert str(v) in nums, (ex.problem, v)
            assert "\n" not in ex.problem and ex.problem.endswith("?")

    def test_distractors(self) -> None:
        for ex in _examples(300, n_distractors_range=(0, 3)):
            n_d = sum(t.startswith("distractor") for t in ex.meta["template_ids"])
            assert n_d == ex.meta["n_distractors"]
        assert {e.meta["n_distractors"] for e in _examples(300)} == {0, 1, 2, 3}

    def test_shapes_and_ops(self) -> None:
        exs = _examples()
        assert {e.meta["shape"] for e in exs} == {"chain", "join"}
        ops = {n["op"] for e in exs for n in e.meta["graph"]}
        assert ops == {"+", "-", "*", "/"}

    def test_diversity(self) -> None:
        exs = _examples()
        tids = {t for e in exs for t in e.meta["template_ids"]}
        names = {e.meta["names"][0] for e in exs}
        items = {e.meta["item"] for e in exs}
        assert len(tids) >= 100, len(tids)
        assert len(names) >= 60, len(names)
        assert len(items) >= 35, len(items)
        firsts = {e.problem.split(" ")[1] for e in exs}
        assert len(firsts) >= 20

    def test_round_trip_through_format(self) -> None:
        for ex in _examples(300):
            text = lsform.render_example(ex)
            assert lsform.parse_answer(text) == ex.answer
            assert lsform.parse_steps(text) == ex.steps
            assert lssco.math_steps_valid(text)


class Test_template_split:
    def test_ids_disjoint(self) -> None:
        train = set(lsmam3.template_ids("train"))
        held = set(lsmam3.template_ids("heldout"))
        assert train and held and not train & held
        assert train | held == set(lsmam3.template_ids("all"))

    def test_examples_respect_split(self) -> None:
        train = set(lsmam3.template_ids("train"))
        held = set(lsmam3.template_ids("heldout"))
        used_train = {t for e in _examples(400) for t in e.meta["template_ids"]}
        used_held = {
            t for e in _examples(400, template_split="heldout") for t in e.meta["template_ids"]
        }
        assert used_train <= train
        assert used_held <= held
        assert not used_train & used_held

    def test_heldout_valid(self) -> None:
        for ex in _examples(300, template_split="heldout"):
            assert all(lssco.check_math_step(s) is True for s in ex.steps)
            assert lsmam3.evaluate_graph(ex.meta["graph"], ex.meta["given"]) == int(ex.answer)


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
            rec = lsmam3.example_to_record(ex)
            assert lsmam3.problem_from_record(rec) == ex.problem

    def test_n_steps_range(self) -> None:
        exs = _examples(400, n_steps_range=(5, 8))
        assert {len(e.steps) for e in exs} == {5, 6, 7, 8}
        exs = _examples(400, n_steps_range=(1, 4))
        assert {len(e.steps) for e in exs} == {1, 2, 3, 4}

    def test_invalid_args(self) -> None:
        rng = random.Random(0)
        with pytest.raises(ValueError):
            lsmam3.generate_example(rng, 0)
        with pytest.raises(ValueError):
            lsmam3.generate_example(rng, 2, template_split="test")
        with pytest.raises(ValueError):
            lsmam3.generate_example(rng, 2, n_distractors=100)
