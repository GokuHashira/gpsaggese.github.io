"""
Unit tests for `logic_slm_scorers.py`.

Import as:

import test.test_logic_slm_scorers as ttlssc
"""

import itertools
import random

import pytest

import logic_slm_math_m1 as lsmam1
import logic_slm_format as lsform
import logic_slm_scorers as lssco


class Test_exact_match:
    def test_basic(self) -> None:
        assert lssco.exact_match("1 + 1 = 2\n#### 2", "2", "math")
        assert lssco.exact_match("#### 02", "2", "math")
        assert not lssco.exact_match("1 + 1 = 2\n#### 3", "2", "math")
        assert not lssco.exact_match("1 + 1 = 2", "2", "math")
        assert lssco.exact_match("#### Yes", "yes", "logic")


class Test_pass_at_k:
    def test_brute_force(self) -> None:
        # Compare with explicit enumeration of k-subsets.
        for n, c, k in [(5, 2, 1), (5, 2, 3), (6, 1, 2), (4, 0, 2), (4, 4, 4)]:
            labels = [1] * c + [0] * (n - c)
            subsets = list(itertools.combinations(labels, k))
            expected = sum(any(s) for s in subsets) / len(subsets)
            assert lssco.pass_at_k(n, c, k) == pytest.approx(expected)

    def test_k1_is_accuracy(self) -> None:
        assert lssco.pass_at_k(10, 3, 1) == pytest.approx(0.3)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            lssco.pass_at_k(5, 6, 1)
        with pytest.raises(ValueError):
            lssco.pass_at_k(5, 2, 6)

    def test_over_problems(self) -> None:
        out = lssco.pass_at_k_over_problems([0, 4], n=4, ks=[1, 4, 8])
        assert out == {"pass@1": pytest.approx(0.5), "pass@4": pytest.approx(0.5)}


class Test_math_steps:
    @pytest.mark.parametrize(
        "step, ok",
        [
            ("17 + 4 = 21", True),
            ("17 + 4 = 22", False),
            ("63 - 12 = 51", True),
            ("3 - 5 = -2", True),
            ("21 * 3 = 63", True),
            ("80 / 5 = 16", True),
            ("7 / 2 = 7/2", True),
            ("7 / 0 = 0", False),
            ("cost of trays: 10 * 50 = 500", True),
            ("cost of trays: 10 * 50 = 600", False),
            ("p, p -> q => q", None),
            ("so the answer is 5", None),
        ],
    )
    def test_check(self, step, ok) -> None:
        assert lssco.check_math_step(step) is ok

    def test_generator_steps_all_valid(self) -> None:
        for ex in lsmam1.generate_dataset(500, 3, n_ops_range=(1, 8)):
            assert lssco.math_steps_valid(lsform.render_solution(ex))

    def test_invalid_completions(self) -> None:
        assert not lssco.math_steps_valid("#### 5")
        assert not lssco.math_steps_valid("2 + 3 = 5\nthen done\n#### 5")


class Test_score_completions:
    def test_metrics(self) -> None:
        comps = [
            "2 + 3 = 5\n#### 5",  # correct, valid
            "2 + 3 = 6\n#### 5",  # correct answer, invalid step
            "2 + 3 = 5\n#### 6",  # wrong answer
            "2 + 3 = 5",  # no answer (its step is valid)
        ]
        out = lssco.score_completions(comps, ["5"] * 4, "math")
        assert out["exact_match"] == pytest.approx(0.5)
        assert out["has_answer"] == pytest.approx(0.75)
        assert out["steps_valid"] == pytest.approx(0.75)  # the no-answer completion has a valid step
        assert out["correct_invalid_steps"] == pytest.approx(0.25)

    def test_wilson(self) -> None:
        lo, hi = lssco.wilson_interval(50, 100)
        assert lo < 0.5 < hi and hi - lo < 0.21


class Test_logic_scoring:
    def test_logic_steps(self) -> None:
        good = "p, p -> q => q\n#### yes"
        bad = "p, p -> q => r\n#### yes"
        english = "Tom is a cat, all cats are mammals => Tom is a mammal\n#### yes"
        assert lssco.logic_steps_valid(good) is True
        assert lssco.logic_steps_valid(bad) is False
        assert lssco.logic_steps_valid(english) is None
        out = lssco.score_completions([good, bad, english], ["yes"] * 3, "logic")
        assert out["exact_match"] == pytest.approx(1.0)
        assert out["steps_checked"] == pytest.approx(2 / 3)
        assert out["steps_valid"] == pytest.approx(0.5)
        assert out["correct_invalid_steps"] == pytest.approx(0.5)
