"""
Unit tests for `logic_slm_format.py`.

Import as:

import test.test_logic_slm_format as ttlsfo
"""

import pytest

import logic_slm_format as lsform

MATH = lsform.Example(
    task="math",
    problem="(17 + 4) * 3 - 12 = ?",
    steps=["17 + 4 = 21", "21 * 3 = 63", "63 - 12 = 51"],
    answer="51",
)
LOGIC = lsform.Example(
    task="logic",
    problem="p -> q. q -> r. p.",
    question="r",
    steps=["p, p -> q => q", "q, q -> r => r"],
    answer="yes",
)


class Test_render:
    def test_math(self) -> None:
        expected = (
            "Problem: (17 + 4) * 3 - 12 = ?\n"
            "Solution:\n"
            "17 + 4 = 21\n21 * 3 = 63\n63 - 12 = 51\n"
            "#### 51"
        )
        assert lsform.render_example(MATH) == expected

    def test_logic(self) -> None:
        expected = (
            "Premises: p -> q. q -> r. p.\n"
            "Question: r?\n"
            "Solution:\n"
            "p, p -> q => q\nq, q -> r => r\n"
            "#### yes"
        )
        assert lsform.render_example(LOGIC) == expected

    def test_prompt_plus_solution(self) -> None:
        for ex in (MATH, LOGIC):
            assert lsform.render_prompt(ex).endswith("Solution:\n")
            assert (
                lsform.render_prompt(ex) + lsform.render_solution(ex)
                == lsform.render_example(ex)
            )

    def test_unknown_with_no_steps(self) -> None:
        ex = lsform.Example(
            task="logic", problem="p -> q.", question="q", steps=[], answer="unknown"
        )
        assert lsform.render_solution(ex) == "#### unknown"


class Test_validation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(task="chem", problem="x", steps=[], answer="1"),
            dict(task="logic", problem="p.", steps=[], answer="yes"),
            dict(task="logic", problem="p.", question="p", steps=[], answer="true"),
            dict(task="math", problem="1+1", steps=["a\nb"], answer="2"),
            dict(task="math", problem="1+1", steps=["#### 2"], answer="2"),
            dict(task="math", problem="1+1", steps=[], answer=" "),
        ],
    )
    def test_invalid(self, kwargs) -> None:
        with pytest.raises(ValueError):
            lsform.Example(**kwargs)


class Test_parse:
    @pytest.mark.parametrize("ex", [MATH, LOGIC])
    def test_round_trip(self, ex) -> None:
        text = lsform.render_example(ex)
        assert lsform.parse_answer(text) == ex.answer
        assert lsform.parse_steps(text) == ex.steps

    def test_bare_completion(self) -> None:
        text = "2 + 3 = 5\n#### 5\n#### 7\ngarbage"
        assert lsform.parse_answer(text) == "5"
        assert lsform.parse_steps(text) == ["2 + 3 = 5"]

    def test_missing_answer(self) -> None:
        text = "Problem: 2 + 3 = ?\nSolution:\n2 + 3 = 5\n"
        assert lsform.parse_answer(text) is None
        assert lsform.parse_steps(text) == ["2 + 3 = 5"]

    def test_empty_answer(self) -> None:
        assert lsform.parse_answer("Solution:\n####   ") is None

    def test_ignores_marker_in_prompt(self) -> None:
        text = "Problem: what is #### ?\nSolution:\n1 + 1 = 2\n#### 2"
        assert lsform.parse_answer(text) == "2"


class Test_normalize:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("51", "51"),
            ("051", "51"),
            ("+51", "51"),
            ("-0", "0"),
            ("-7", "-7"),
            ("6/8", "3/4"),
            ("4/2", "2"),
            ("-3/4", "-3/4"),
            ("1,234", "1234"),
            ("+1,234", "1234"),
            ("-1,234", "-1234"),
            ("1/0", "1/0"),
            (" 12  apples ", "12 apples"),
        ],
    )
    def test_math(self, raw, expected) -> None:
        assert lsform.normalize_answer(raw, "math") == expected

    @pytest.mark.parametrize("raw", ["Yes", "yes.", " YES "])
    def test_logic(self, raw) -> None:
        assert lsform.normalize_answer(raw, "logic") == "yes"

    def test_none(self) -> None:
        assert lsform.normalize_answer(None, "math") is None

    def test_answers_match(self) -> None:
        assert lsform.answers_match("051", "51", "math")
        assert lsform.answers_match("6/8", "3/4", "math")
        assert not lsform.answers_match("52", "51", "math")
        assert not lsform.answers_match(None, "51", "math")
        assert lsform.answers_match("Unknown", "unknown", "logic")
