"""
Unit tests for `logic_slm_realdata.py` and the v2 answer normalization in
`logic_slm_format.py`.

Tests on real rows are skipped when the raw files are not downloaded.

Import as:

import test.test_logic_slm_realdata as ttlsre
"""

import gzip
import json
import os

import pytest

import logic_slm_format as lsform
import logic_slm_realdata as lsreda

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(_ROOT, "data", "raw")


def _have(rel: str) -> bool:
    import glob

    return bool(glob.glob(os.path.join(RAW, rel)))


def _check_record(rec: dict, gen: str) -> None:
    assert rec["format"] == lsform.FORMAT_VERSION == "v2"
    assert rec["task"] == "math" and rec["gen"] == gen and rec["level"] == gen
    assert rec["prompt"].startswith("Problem: ") and rec["prompt"].endswith("\nSolution:\n")
    assert rec["prompt"].count("\n") == 2
    text = rec["prompt"] + rec["solution"]
    assert lsform.parse_answer(text) == rec["answer"]
    assert rec["solution"].split("\n")[-1] == f"#### {rec['answer']}"
    assert sum(line.startswith("####") for line in rec["solution"].split("\n")) == 1
    assert "\\boxed" not in rec["solution"]


# #############################################################################
# v2 answer normalization
# #############################################################################


class Test_normalize_v2:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            # decimals compare by value
            ("3.50", "7/2"),
            ("3.5", "7/2"),
            ("7/2", "7/2"),
            ("14/4", "7/2"),
            (".5", "1/2"),
            ("3.0", "3"),
            ("-0.25", "-1/4"),
            # thousands separators
            ("1,234", "1234"),
            ("1,234,567", "1234567"),
            ("1,234.50", "2469/2"),
            # currency / percent / units
            ("$5", "5"),
            ("$1,200", "1200"),
            ("-$5", "-5"),
            ("$-5", "-5"),
            ("25%", "25"),
            ("25\\%", "25"),
            ("12 cm", "12"),
            ("45 dollars", "45"),
            ("3 square feet", "3"),
            ("90^\\circ", "90"),
            ("90^{\\circ}", "90"),
            ("18.", "18"),
            ("x = 5", "5"),
            # LaTeX wrappers
            ("\\boxed{5}", "5"),
            ("\\boxed{\\frac{7}{2}}", "7/2"),
            ("$5$", "5"),
            ("$\\boxed{18}$", "18"),
            ("\\frac{3}{4}", "3/4"),
            ("\\dfrac{6}{8}", "3/4"),
            ("\\tfrac{1}{2}", "1/2"),
            ("\\frac12", "1/2"),
            ("-\\frac{1}{3}", "-1/3"),
            ("\\text{5 cm}", "5"),
            ("\\$32", "32"),
            ("−7", "-7"),
            # non-numeric: LaTeX-light canonical string
            ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})"),
            ("$[2, 5)$", "[2,5)"),
            ("\\boxed{ (-\\infty, 0] }", "(-\\infty,0]"),
            ("\\text{(C)}", "(C)"),
            ("2\\sqrt{3}", "2\\sqrt{3}"),
            ("12 apples", "12 apples"),
            ("1/0", "1/0"),
        ],
    )
    def test_math(self, raw, expected) -> None:
        assert lsform.normalize_answer(raw, "math") == expected

    @pytest.mark.parametrize(
        "a, b",
        [
            ("3.50", "3.5"),
            ("3.5", "7/2"),
            ("\\frac{7}{2}", "3.5"),
            ("$1,000", "1000"),
            ("\\boxed{42}", "42"),
            ("\\left[ 2,5 \\right)", "[2, 5)"),
            ("\\dfrac{\\pi}{4}", "\\frac{\\pi}{4}"),
        ],
    )
    def test_match(self, a, b) -> None:
        assert lsform.answers_match(a, b, "math")

    @pytest.mark.parametrize(
        "a, b",
        [("3.51", "3.5"), ("12 apples", "12"), ("[2,5)", "[2,5]"), ("1/3", "0.33")],
    )
    def test_no_match(self, a, b) -> None:
        assert not lsform.answers_match(a, b, "math")

    def test_v1_rendering_unchanged(self) -> None:
        ex = lsform.Example(task="math", problem="2 + 3 = ?", steps=["2 + 3 = 5"], answer="5")
        assert lsform.render_example(ex) == "Problem: 2 + 3 = ?\nSolution:\n2 + 3 = 5\n#### 5"


# #############################################################################
# Normalizers on hand-written rows
# #############################################################################


class Test_gsm8k:
    def test_row(self) -> None:
        row = {
            "question": "Natalia sold clips to 48 friends.\nThen half as many in May. How many?",
            "answer": "In May she sold 48/2 = <<48/2=24>>24 clips.\n"
            "In total 48+24 = <<48+24=72>>72 clips.\n#### 72",
        }
        ex, reason = lsreda.normalize_gsm8k(row)
        assert reason is None
        assert ex.problem == "Natalia sold clips to 48 friends. Then half as many in May. How many?"
        assert ex.steps == ["In May she sold 48/2 = 24 clips.", "In total 48+24 = 72 clips."]
        assert ex.answer == "72"
        rec = lsreda.example_to_record(ex, "gsm8k-train")
        assert rec["solution"] == "In May she sold 48/2 = 24 clips.\nIn total 48+24 = 72 clips.\n#### 72"
        assert rec["gen"] == "gsm8k-train" and rec["level"] == "gsm8k-train"

    def test_answer_commas(self) -> None:
        ex, _ = lsreda.normalize_gsm8k({"question": "q?", "answer": "a\n#### 1,000"})
        assert ex.answer == "1000"

    def test_no_marker(self) -> None:
        assert lsreda.normalize_gsm8k({"question": "q?", "answer": "no answer"}) == (None, "bad_answer")


class Test_openmathinstruct2:
    def _row(self, sol, ans="18", src="augmented_gsm8k"):
        return {"problem": "How much?", "generated_solution": sol, "expected_answer": ans, "problem_source": src}

    def test_strip_boxed_line(self) -> None:
        sol = "She eats 3 eggs.\nSo she sells 16 - 3 = 13 eggs and earns 13 * 2 = 26.\n\nThus, she earns \\boxed{18} dollars."
        ex, reason = lsreda.normalize_openmathinstruct2(self._row(sol))
        assert reason is None
        assert ex.steps == ["She eats 3 eggs.", "So she sells 16 - 3 = 13 eggs and earns 13 * 2 = 26."]
        assert ex.answer == "18"
        assert ex.meta == {"level": "openmathinstruct2", "problem_source": "augmented_gsm8k"}

    def test_strip_boxed_sentence_keeps_rest(self) -> None:
        sol = "Step one is 9.\nAdding gives 18. So the answer is $\\boxed{18}$."
        ex, _ = lsreda.normalize_openmathinstruct2(self._row(sol))
        assert ex.steps == ["Step one is 9.", "Adding gives 18."]

    def test_bare_boxed_line(self) -> None:
        ex, _ = lsreda.normalize_openmathinstruct2(self._row("We get 2 + 16 = 18.\n\\[ \\boxed{18} \\]"))
        assert ex.steps == ["We get 2 + 16 = 18."]

    def test_latex_answer(self) -> None:
        ex, _ = lsreda.normalize_openmathinstruct2(self._row("Solve it.\nAnswer: \\boxed{\\frac{1}{2}}", ans="\\frac{1}{2}", src="math"))
        assert ex.answer == "\\frac{1}{2}"
        assert lsform.answers_match("0.5", ex.answer, "math")

    def test_gsm8k_only(self) -> None:
        row = self._row("x\n\\boxed{18}", src="augmented_math")
        assert lsreda.normalize_openmathinstruct2(row, gsm8k_only=True) == (None, "not_gsm8k_source")
        assert lsreda.normalize_openmathinstruct2(row)[0] is not None

    def test_drops(self) -> None:
        assert lsreda.normalize_openmathinstruct2(self._row("\\boxed{18}"))[1] == "empty_solution"
        assert lsreda.normalize_openmathinstruct2(self._row("x\n\\boxed{}", ans=""))[1] == "bad_answer"
        long_sol = " ".join(["word"] * 1600) + "\n\\boxed{18}"
        assert lsreda.normalize_openmathinstruct2(self._row(long_sol))[1] == "too_long"
        junk = "解" * 50 + " 18\n\\boxed{18}"
        assert lsreda.normalize_openmathinstruct2(self._row(junk))[1] == "non_ascii"


class Test_metamathqa:
    def test_row(self) -> None:
        row = {"query": "What is 2+3?", "response": "We add 2 and 3.\n2 + 3 = 5.\nThe answer is: 5", "type": "GSM_AnsAug"}
        ex, reason = lsreda.normalize_metamathqa(row)
        assert reason is None
        assert ex.steps == ["We add 2 and 3.", "2 + 3 = 5."] and ex.answer == "5"
        assert ex.meta == {"level": "metamathqa", "type": "GSM_AnsAug"}

    def test_inline_and_latex(self) -> None:
        row = {"query": "q", "response": "So x is 1/2. The answer is: \\frac{1}{2}", "type": "MATH_SV"}
        ex, _ = lsreda.normalize_metamathqa(row)
        assert ex.steps == ["So x is 1/2."] and ex.answer == "\\frac{1}{2}"

    def test_missing(self) -> None:
        assert lsreda.normalize_metamathqa({"query": "q", "response": "no final"})[1] == "bad_answer"
        assert lsreda.normalize_metamathqa({"query": "q", "response": "x\nThe answer is: "})[1] == "bad_answer"


class Test_orcamath:
    @pytest.mark.parametrize(
        "text, answer",
        [
            ("First 3 + 4 = 7.\nTherefore, the number is 7.", "7"),
            ("Cost is 5 * 3.\nSo the total cost is $1,250.", "1250"),
            ("Some steps.\nThe 3 friends spent $45.50 in total, which is 45.50.", "45.50"),
            ("Steps.\nThus the answer is \\boxed{12}.", "12"),
            ("Steps.\nThe speed is approximately 3.33 km/h.", "3.33"),
            ("Steps.\nSo, John has **15** apples left.", "15"),
        ],
    )
    def test_extract(self, text, answer) -> None:
        ex, reason = lsreda.normalize_orcamath({"question": "q?", "answer": text})
        assert reason is None, text
        assert ex.answer == answer
        assert "**" not in "\n".join(ex.steps)

    @pytest.mark.parametrize(
        "text, reason",
        [
            ("Steps.\nTherefore, it takes 3 hours and 20 minutes.", "ambiguous_final_answer"),
            ("Steps 1 and 2.\nSo we cannot tell.", "no_final_answer"),
            ("", "no_final_answer"),
        ],
    )
    def test_drop(self, text, reason) -> None:
        assert lsreda.normalize_orcamath({"question": "q?", "answer": text}) == (None, reason)


class Test_clean_answer:
    @pytest.mark.parametrize(
        "raw, expected",
        [("18", "18"), (" $1,000. ", "1000"), ("$\\frac{1}{2}$", "\\frac{1}{2}"), ("\\boxed{7}", "7"), ("", None), ("a\nb", None), ("x" * 200, None)],
    )
    def test_values(self, raw, expected) -> None:
        assert lsreda.clean_answer(raw) == expected


class Test_proofwriter_record:
    def test_labels(self) -> None:
        base = {"theory": "The cat is red. If something is red then it is big.", "question": "The cat is big."}
        for label, verdict in (("True", "yes"), ("False", "no"), ("Unknown", "unknown")):
            rec = lsreda.proofwriter_record({**base, "answer": label}, 0)
            assert rec["answer"] == verdict and rec["task"] == "logic"
            assert rec["prompt"] == (
                "Premises: The cat is red. If something is red then it is big.\n"
                "Question: The cat is big?\nSolution:\n"
            )
        assert lsreda.proofwriter_record({**base, "answer": "maybe"}, 0) is None


class Test_build_train_metamath(object):
    def test_shards_and_stats(self, tmp_path) -> None:
        raw = tmp_path / "raw" / "metamathqa"
        raw.mkdir(parents=True)
        rows = [{"query": f"What is {i}+1?", "response": f"{i} + 1 = {i + 1}.\nThe answer is: {i + 1}", "type": "t"} for i in range(25)]
        rows.append({"query": "bad", "response": "nothing", "type": "t"})
        (raw / "MetaMathQA-395K.json").write_text(json.dumps(rows))
        out = tmp_path / "out"
        st = lsreda.build_train("metamathqa", str(out), raw_dir=str(tmp_path / "raw"), workers=2, chunk=10)
        assert st == {"in": 26, "kept": 25, "drop:bad_answer": 1}
        shards = sorted(out.glob("part-*.jsonl.gz"))
        assert len(shards) == 3
        recs = [json.loads(line) for p in shards for line in gzip.open(p, "rt")]
        assert len(recs) == 25
        for r in recs:
            _check_record(r, "metamathqa")


# #############################################################################
# Real rows (skipped if not downloaded)
# #############################################################################


def _first_rows(source: str, n: int = 2000):
    if source == "metamathqa":
        with open(os.path.join(RAW, "metamathqa/MetaMathQA-395K.json")) as f:
            return json.load(f)[:n]
    pattern, cols = lsreda._source_spec(source, RAW)
    path = lsreda._parquet_files(pattern)[0]
    rows = []
    for r in lsreda.iter_parquet(path, cols, 0):
        rows.append(r)
        if len(rows) >= n:
            break
    return rows


_REAL = [
    ("gsm8k-train", "gsm8k/main/train-*.parquet", 0.99),
    ("openmathinstruct2", "openmathinstruct2/data/train_2M-*.parquet", 0.95),
    ("metamathqa", "metamathqa/MetaMathQA-395K.json", 0.95),
    ("orcamath", "orcamath/data/train-*.parquet", 0.5),
]


@pytest.mark.parametrize("source, rel, min_keep", _REAL)
def test_real_rows(source, rel, min_keep) -> None:
    if not _have(rel):
        pytest.skip(f"{rel} not downloaded")
    rows = _first_rows(source)
    recs, st = lsreda.normalize_rows(rows, source)
    assert st["in"] == len(rows)
    assert st["kept"] / st["in"] >= min_keep, st
    for r in recs:
        _check_record(r, source)
        assert lsform.normalize_answer(r["answer"], "math")


_EVAL = [
    ("gsm8k-test", "gsm8k/main/test-*.parquet", 1319),
    ("math500", "math500/test.jsonl", 500),
    ("svamp", "svamp/data/test-*.parquet", 300),
    ("proofwriter", "proofwriter/data/test-*.parquet", None),
]


@pytest.mark.parametrize("name, rel, count", _EVAL)
def test_eval_loaders(name, rel, count) -> None:
    if not _have(rel):
        pytest.skip(f"{rel} not downloaded")
    recs = lsreda.EVAL_LOADERS[name](RAW)
    if count is not None:
        assert len(recs) == count
    assert len(recs) > 0
    for r in recs:
        assert r["gen"] == name and r["answer"]
        assert r["prompt"].endswith("\nSolution:\n")
        if r["solution"]:
            assert lsform.parse_answer(r["prompt"] + r["solution"]) == r["answer"]
    if name == "proofwriter":
        assert {r["answer"] for r in recs} == {"yes", "no", "unknown"}
        assert all(r["task"] == "logic" for r in recs)


# #############################################################################
# Loaders on hand-written files mimicking each source's schema
# #############################################################################


def _parquet(path, rows) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), str(path), row_group_size=2)


@pytest.fixture
def fake_raw(tmp_path):
    raw = tmp_path / "raw"
    gsm = [
        {"question": f"Tom has {i} apples and buys 2 more. How many now?", "answer": f"He has {i}+2=<<{i}+2={i + 2}>>{i + 2} apples.\n#### {i + 2}"}
        for i in range(5)
    ]
    _parquet(raw / "gsm8k/main/train-00000-of-00001.parquet", gsm)
    _parquet(raw / "gsm8k/main/test-00000-of-00001.parquet", gsm[:3])
    omi = [
        {"problem": "What is 9+9?", "generated_solution": "We add 9 and 9.\nThe answer is \\boxed{18}.", "expected_answer": "18", "problem_source": "augmented_gsm8k"},
        {"problem": "Solve 2x=1.", "generated_solution": "Divide by 2.\n$x = \\boxed{\\frac{1}{2}}$", "expected_answer": "\\frac{1}{2}", "problem_source": "augmented_math"},
        {"problem": "Empty?", "generated_solution": "\\boxed{3}", "expected_answer": "3", "problem_source": "math"},
    ]
    _parquet(raw / "openmathinstruct2/data/train_2M-00000-of-00002.parquet", omi[:2])
    _parquet(raw / "openmathinstruct2/data/train_2M-00001-of-00002.parquet", omi[2:])
    orca = [
        {"question": "A shop sells 4 pens at $3 each. Total?", "answer": "4 * 3 = 12.\nSo the total cost is $12."},
        {"question": "How long?", "answer": "It takes 3 hours and 20 minutes."},
    ]
    _parquet(raw / "orcamath/data/train-00000-of-00001.parquet", orca)
    (raw / "math500").mkdir(parents=True)
    with open(raw / "math500/test.jsonl", "w") as f:
        for row in [
            {"problem": "Find the interval.\nGive it in LaTeX.", "solution": "...", "answer": "[2, 5)", "subject": "Algebra", "level": 3, "unique_id": "test/algebra/1.json"},
            {"problem": "Compute $\\frac{1}{2}+\\frac{1}{2}$.", "solution": "...", "answer": "1", "subject": "Prealgebra", "level": 1, "unique_id": "test/prealgebra/2.json"},
        ]:
            f.write(json.dumps(row) + "\n")
    _parquet(
        raw / "svamp/data/test-00000-of-00001.parquet",
        [
            {"ID": "chal-1", "Body": "Jack had 8 pens.", "Question": "How many pens after he gave away 3?", "Equation": "( 8.0 - 3.0 )", "Answer": 5.0, "Type": "Subtraction"},
            {"ID": "chal-2", "Body": "A cake is cut into 4 parts.", "Question": "What fraction is one part?", "Equation": "( 1.0 / 4.0 )", "Answer": 0.25, "Type": "Common-Division"},
        ],
    )
    _parquet(
        raw / "proofwriter/data/test-00000-of-00001.parquet",
        [
            {"id": "p1", "theory": "The cat is red. Red things are big.", "question": "The cat is big.", "answer": "True", "QDep": 1},
            {"id": "p2", "theory": "The cat is red.", "question": "The cat is blue.", "answer": "Unknown", "QDep": 0},
            {"id": "p3", "theory": "The cat is not blue.", "question": "The cat is blue.", "answer": "False", "QDep": 0},
            {"id": "p4", "theory": "x", "question": "y", "answer": "", "QDep": 0},
        ],
    )
    return raw


class Test_fake_sources:
    def test_build_train_all(self, fake_raw, tmp_path) -> None:
        expected = {
            "gsm8k-train": {"in": 5, "kept": 5},
            "openmathinstruct2": {"in": 3, "kept": 2, "drop:empty_solution": 1},
            "orcamath": {"in": 2, "kept": 1, "drop:ambiguous_final_answer": 1},
        }
        for source, exp in expected.items():
            out = tmp_path / source
            st = lsreda.build_train(source, str(out), raw_dir=str(fake_raw), workers=2)
            assert st == exp, source
            recs = [json.loads(line) for p in sorted(out.glob("part-*.jsonl.gz")) for line in gzip.open(p, "rt")]
            assert len(recs) == exp["kept"]
            for r in recs:
                _check_record(r, source)
        st = lsreda.build_train("openmathinstruct2", str(tmp_path / "omi-gsm"), raw_dir=str(fake_raw), gsm8k_only=True)
        assert st == {"in": 3, "kept": 1, "drop:not_gsm8k_source": 2}

    def test_eval_sets(self, fake_raw, tmp_path) -> None:
        counts = lsreda.build_eval(str(tmp_path / "eval"), raw_dir=str(fake_raw))
        assert counts == {"gsm8k-test": 3, "math500": 2, "svamp": 2, "proofwriter": 3}
        load = lambda n: [json.loads(line) for line in open(tmp_path / "eval" / f"{n}.jsonl")]
        gsm = load("gsm8k-test")
        assert gsm[0]["solution"] == "He has 0+2=2 apples.\n#### 2" and gsm[0]["answer"] == "2"
        m5 = load("math500")
        assert m5[0]["prompt"] == "Problem: Find the interval. Give it in LaTeX.\nSolution:\n"
        assert m5[0]["answer"] == "[2, 5)" and m5[0]["solution"] == ""
        sv = load("svamp")
        assert sv[0]["prompt"] == "Problem: Jack had 8 pens. How many pens after he gave away 3?\nSolution:\n"
        assert [r["answer"] for r in sv] == ["5", "0.25"]
        pw = load("proofwriter")
        assert [r["answer"] for r in pw] == ["yes", "unknown", "no"]
        assert pw[0]["prompt"] == "Premises: The cat is red. Red things are big.\nQuestion: The cat is big?\nSolution:\n"
        assert pw[0]["qdep"] == 1


def test_unbox_all() -> None:
    import logic_slm_realdata as lsreal

    assert lsreal.unbox_all(r"The points are $\boxed{\sqrt{5}}$ units apart.") == r"The points are $\sqrt{5}$ units apart."
    assert lsreal.unbox_all(r"x = \boxed{\frac{1}{2}} and y = \boxed{3}") == r"x = \frac{1}{2} and y = 3"
    assert lsreal.unbox_all("no box here") == "no box here"
    assert lsreal.unbox_all(r"broken \boxed{3") == r"broken \boxed{3"
