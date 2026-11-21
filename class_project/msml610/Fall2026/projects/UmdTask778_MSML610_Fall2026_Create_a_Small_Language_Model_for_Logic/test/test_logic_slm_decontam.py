"""
Unit tests for `logic_slm_decontam.py`.

Import as:

import test.test_logic_slm_decontam as ttlsde
"""

import gzip
import json
import subprocess
import sys

import logic_slm_decontam as lsdeco

EVAL_Q = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
    "morning and bakes muffins for her friends every day with four. How much "
    "in dollars does she make every day at the farmers' market?"
)


class Test_tokenize:
    def test_basic(self) -> None:
        assert lsdeco.tokenize("Janet's ducks, lay 16-eggs!") == ["janet", "s", "ducks", "lay", "16", "eggs"]

    def test_stable_hash(self) -> None:
        # Fixed value: blake2b does not depend on PYTHONHASHSEED.
        h = lsdeco.ngram_hashes("a b c", n=3)
        code = "import logic_slm_decontam as d; print(sorted(d.ngram_hashes('a b c', n=3)))"
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": "123", "PYTHONPATH": "."},
            check=True,
        ).stdout
        assert out.strip() == str(sorted(h))


class Test_contamination:
    index = lsdeco.build_index([EVAL_Q])

    def test_planted_overlap(self) -> None:
        # 13 consecutive eval words embedded in other text (case/punct differ).
        planted = "Here is a new task. SHE EATS THREE for breakfast, every morning and bakes muffins for her friends every day! Then more."
        assert lsdeco.is_contaminated(planted, self.index)

    def test_exact_copy(self) -> None:
        assert lsdeco.is_contaminated(EVAL_Q, self.index)

    def test_near_miss(self) -> None:
        # Only 12 consecutive eval words.
        near = "Bob says: she eats three for breakfast every morning and bakes muffins for her cousins."
        assert not lsdeco.is_contaminated(near, self.index)
        # A changed word in every 13-word window.
        changed = EVAL_Q.replace("breakfast", "lunch").replace("friends", "kids").replace("dollars", "euros").replace("16", "15")
        changed = changed.replace("market", "stall").replace("bakes", "makes")
        assert not lsdeco.is_contaminated(changed, self.index)

    def test_short_texts(self) -> None:
        idx = lsdeco.build_index(["What is 2 plus 3?"])
        assert lsdeco.is_contaminated("what is 2 plus 3", idx)
        assert not lsdeco.is_contaminated("What is 2 plus 3 times 4?", idx)
        assert not lsdeco.is_contaminated("", idx)

    def test_strip_prompt(self) -> None:
        assert lsdeco.strip_prompt("Problem: abc\nSolution:\n") == "abc"
        assert lsdeco.strip_prompt("Premises: p.\nQuestion: q?\nSolution:\n") == "p.\nq?"


def _write_shard(path, recs) -> None:
    with gzip.open(path, "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_filter_dir_and_cli(tmp_path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "gsm8k-test.jsonl").write_text(
        json.dumps({"prompt": f"Problem: {EVAL_Q}\nSolution:\n", "solution": "", "answer": "18"}) + "\n"
    )
    src = tmp_path / "src"
    src.mkdir()
    clean = [{"prompt": f"Problem: Tom has {i} apples and buys some more apples today at the shop near his home.\nSolution:\n", "solution": "#### 1", "gen": "x"} for i in range(50)]
    dirty = {"prompt": "Problem: Something new.\nSolution:\n", "solution": f"Copied: {EVAL_Q}\n#### 18", "gen": "x"}
    _write_shard(src / "part-00000.jsonl.gz", clean[:25] + [dirty])
    _write_shard(src / "part-00001.jsonl.gz", clean[25:])
    index, counts = lsdeco.load_eval_index(str(eval_dir))
    assert counts == {"gsm8k-test.jsonl": 1}
    out = tmp_path / "out"
    st = lsdeco.filter_dir(str(src), str(out), index, workers=2)
    assert st == {"in": 51, "kept": 50, "removed": 1, "removed:x": 1}
    kept = [json.loads(line) for p in sorted(out.glob("*.jsonl.gz")) for line in gzip.open(p, "rt")]
    assert kept == clean
    # CLI end to end.
    out2 = tmp_path / "out2"
    report = tmp_path / "report.json"
    subprocess.run(
        [sys.executable, "logic_slm_decontam.py", "--in-dir", str(src), "--out-dir", str(out2), "--eval-dir", str(eval_dir), "--workers", "2", "--report", str(report)],
        check=True,
        capture_output=True,
    )
    rep = json.loads(report.read_text())
    assert rep["sources"][str(src)]["removed"] == 1
