"""
Unit tests for `logic_slm_tinygsm.py`.

Import as:

import test.test_logic_slm_tinygsm as ttlsmtgsm
"""

import fractions
import os
import textwrap

import pytest

import logic_slm_format as lsform
import logic_slm_scorers as lssco
import logic_slm_tinygsm as lsmtgsm

_SHARD0 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "tinygsm", "data", "train-00000-of-00017.parquet",
)
_Q = "Tom has 12 cookies and eats 4. How many are left?"


def _code(body: str) -> str:
    """
    Wrap a function body in the TinyGSM function template.
    """
    body = textwrap.indent(textwrap.dedent(body).strip(), "    ")
    return f'def simple_math_problem() -> int:\n    """{_Q}"""\n{body}\n'


def _convert(body: str):
    return lsmtgsm.convert_row_with_reason(_Q, _code(body))


def _value(step: str) -> fractions.Fraction:
    return fractions.Fraction(step.rsplit("=", 1)[1].strip())


def _check_example(ex: lsform.Example) -> None:
    """
    Every step checks, the last step computes the answer, and the render /
    parse round trip is lossless.
    """
    assert ex.steps and ex.meta["n_steps"] == len(ex.steps)
    for s in ex.steps:
        assert lssco.check_math_step(s) is True, s
        assert "." not in s.rsplit(":", 1)[1], s
    assert _value(ex.steps[-1]) == fractions.Fraction(ex.answer)
    assert lsform.normalize_answer(ex.answer, "math") == ex.answer
    text = lsform.render_example(ex)
    assert lsform.parse_steps(text) == ex.steps
    assert lsform.parse_answer(text) == ex.answer
    assert lssco.math_steps_valid(lsform.render_solution(ex))


class Test_convert_row:
    def test_simple_chain(self) -> None:
        ex, reason = _convert(
            """
            cookies = 12
            eaten = 4
            left = cookies - eaten
            result = left
            return result
            """
        )
        assert reason is None
        assert ex.steps == ["left: 12 - 4 = 8"]
        assert ex.answer == "8"
        assert ex.problem == _Q
        assert ex.meta == {"level": "tinygsm", "n_steps": 1}
        assert lsform.render_example(ex) == f"Problem: {_Q}\nSolution:\nleft: 12 - 4 = 8\n#### 8"
        _check_example(ex)

    def test_nested_expression(self) -> None:
        ex, reason = _convert(
            """
            price_per_box = 3
            boxes = 4
            fee = 5
            total_cost = (price_per_box * boxes + fee) * 2 - 6
            return total_cost
            """
        )
        assert reason is None
        assert ex.steps == [
            "total cost: 3 * 4 = 12",
            "total cost: 12 + 5 = 17",
            "total cost: 17 * 2 = 34",
            "total cost: 34 - 6 = 28",
        ]
        assert ex.answer == "28"
        _check_example(ex)

    def test_division_integer_and_fraction_steps(self) -> None:
        ex, reason = _convert(
            """
            total = 7
            half = total / 2
            doubled = half * 4
            per_day = doubled // 7
            return per_day
            """
        )
        assert reason is None
        assert ex.steps == [
            "half: 7 / 2 = 7/2",
            "doubled: 7/2 * 4 = 14",
            "per day: 14 / 7 = 2",
        ]
        _check_example(ex)

    def test_float_literals_are_exact(self) -> None:
        ex, reason = _convert(
            """
            price = 2.5
            discount = 0.2
            n = 10
            cost = price * n * (1 - discount)
            return cost
            """
        )
        assert reason is None
        assert ex.steps == [
            "cost: 5/2 * 10 = 25",
            "cost: 1 - 1/5 = 4/5",
            "cost: 25 * 4/5 = 20",
        ]
        _check_example(ex)

    def test_augassign_pow_sum_and_pruning(self) -> None:
        ex, reason = _convert(
            """
            side = 3
            area = side ** 2
            unused = side + 100
            total = sum([area, side, 1])
            total += 2
            return int(total)
            """
        )
        assert reason is None
        assert ex.steps == [
            "area: 3 * 3 = 9",
            "total: 9 + 3 = 12",
            "total: 12 + 1 = 13",
            "total: 13 + 2 = 15",
        ]
        _check_example(ex)

    def test_min_max_no_step(self) -> None:
        ex, reason = _convert(
            """
            a = 3 + 4
            b = 10
            c = max(a, b) * 2
            return c
            """
        )
        assert reason is None
        assert ex.steps == ["a: 3 + 4 = 7", "c: 10 * 2 = 20"]
        _check_example(ex)

    def test_return_expression(self) -> None:
        ex, reason = _convert(
            """
            a = 5
            b = 6
            return a * b
            """
        )
        assert reason is None
        assert ex.steps == ["result: 5 * 6 = 30"]

    @pytest.mark.parametrize(
        "body,reason",
        [
            ("a = 7\nb = a / 2\nreturn b", "non_integer_answer"),
            ("a = 7\nb = a // 2\nreturn b", "floordiv_inexact"),
            ("a = 7.5\nreturn round(a)", "lossy_round"),
            ("a = 4\nb = 0\nreturn a / b", "div_by_zero"),
            ("a = 12\nreturn a", "no_steps"),
            ("a = 10 ** 6\nb = a * a\nreturn b", "too_large"),
            ("a = 2\nreturn a ** 50", "pow"),
            ("a = 7\nreturn a % 2", "binop:Mod"),
            ("a = 1\nb = 2\nc = a + b\nd = c * 5\nreturn min(c, d)", "answer_not_last_step"),
            ("return x + 1", "unbound_name"),
        ],
    )
    def test_dropped(self, body: str, reason: str) -> None:
        assert _convert(body) == (None, reason)

    @pytest.mark.parametrize(
        "body,reason",
        [
            ("total = 0\nfor i in range(3):\n    total += i\nreturn total", "stmt:For"),
            ("a = 5\nwhile a > 0:\n    a -= 1\nreturn a", "stmt:While"),
            ("a = 5\nif a > 3:\n    a = a + 1\nreturn a", "stmt:If"),
            ("import os\nreturn 1 + 1", "stmt:Import"),
            ("a = 2\nb = os.system('ls') + a\nreturn b", "call"),
            ("a = 2\nb = math.pi * a\nreturn b", "expr:Attribute"),
            ("a = len('abc') + 1\nreturn a", "call"),
            ("a = 'ab' + 'c'\nreturn a", "expr:Constant"),
            ("a = [i for i in range(3)]\nreturn a", "expr:ListComp"),
            ("a = 3\nb = a if a > 1 else 2\nreturn b + 1", "expr:IfExp"),
            ("try:\n    a = 1\nexcept Exception:\n    a = 2\nreturn a + 1", "stmt:Try"),
            ("a = True + 1\nreturn a", "expr:Constant"),
            ("a, b = 1, 2\nreturn a + b", "assign_target"),
            ("a = 1 + 2\nreturn a\nb = 3", "code_after_return"),
        ],
    )
    def test_forbidden_constructs(self, body: str, reason: str) -> None:
        assert _convert(body) == (None, reason)

    def test_not_a_simple_function(self) -> None:
        conv = lsmtgsm.convert_row_with_reason
        assert conv(_Q, "import os\n" + _code("return 1 + 2")) == (None, "not_single_function")
        assert conv(_Q, "def f(x):\n    return x + 1\n") == (None, "has_args")
        assert conv(_Q, "def f(:\n") == (None, "parse_error")
        assert conv(_Q, "x" * 5000) == (None, "too_long")
        assert conv(_Q, None) == (None, "missing_field")
        assert conv("   ", _code("return 1 + 2")) == (None, "empty_question")
        assert lsmtgsm.convert_row(_Q, _code("return 1 + 2")).answer == "3"

    def test_malicious_code_never_executed(self, tmp_path) -> None:
        marker = tmp_path / "pwned.txt"
        payloads = [
            f"open({str(marker)!r}, 'w').write('x')\nreturn 1 + 1",
            f"a = 1 + open({str(marker)!r}, 'w').write('x')\nreturn a",
            f"a = __import__('os').system('touch {marker}')\nreturn a + 1",
            f"a = 2 * eval(\"open({str(marker)!r}, 'w').write('x')\")\nreturn a",
            f"exec(\"open({str(marker)!r}, 'w')\")\nreturn 2 + 2",
            f"a = sum([open({str(marker)!r}, 'w').write('x'), 1])\nreturn a + 1",
        ]
        for body in payloads:
            ex, reason = _convert(body)
            assert ex is None and reason is not None, body
        # Module-level code outside the function is rejected too.
        code = f"open({str(marker)!r}, 'w').write('x')\n" + _code("return 1 + 1")
        assert lsmtgsm.convert_row(_Q, code) is None
        assert not marker.exists()

    def test_record(self) -> None:
        ex = lsmtgsm.convert_row(_Q, _code("a = 12\nb = 4\nreturn a - b"))
        rec = lsmtgsm.example_to_record(ex)
        assert rec == {
            "gen": "tinygsm",
            "task": "math",
            "format": lsform.FORMAT_VERSION,
            "prompt": f"Problem: {_Q}\nSolution:\n",
            "solution": "result: 12 - 4 = 8\n#### 8",
            "answer": "8",
            "n_steps": 1,
        }


class Test_format_number:
    def test_values(self) -> None:
        F = fractions.Fraction
        assert lsmtgsm.format_number(F(-3)) == "-3"
        assert lsmtgsm.format_number(F(14, 4)) == "7/2"
        assert lsmtgsm.format_number(F(-1, 3)) == "-1/3"
        assert lsmtgsm.name_to_words("total_cost") == "total cost"
        assert lsmtgsm.name_to_words("numApples") == "num apples"
        assert lsmtgsm.name_to_words("_") == "value"


@pytest.mark.skipif(not os.path.exists(_SHARD0), reason="TinyGSM shard 0 not downloaded")
class Test_real_rows:
    def test_first_rows(self, tmp_path) -> None:
        n = 2000
        out = str(tmp_path / "shard0.jsonl.gz")
        st = lsmtgsm.convert_shard(_SHARD0, out, batch_size=500, limit=n)
        assert st["rows_in"] == n
        rate = st["kept"] / n
        print(f"\nTinyGSM shard 0, first {n} rows: kept {st['kept']} ({rate:.1%}); drops {st['drops']}")
        assert rate >= 0.5, st
        import pyarrow.parquet as pq

        batch = next(pq.ParquetFile(_SHARD0).iter_batches(batch_size=n, columns=["question", "code"]))
        cols = batch.to_pydict()
        kept = 0
        for q, c in zip(cols["question"], cols["code"]):
            ex = lsmtgsm.convert_row(q, c)
            if ex is not None:
                kept += 1
                _check_example(ex)
        assert kept == st["kept"]
