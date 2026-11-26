import json
import os

import numpy as np
import pytest

import confidence_intervals as ci


class Test_bootstrap_ci:
    def test_interval_contains_mean_and_is_ordered(self) -> None:
        v = np.array([1] * 300 + [0] * 700)
        m, lo, hi = ci.bootstrap_ci(v, n_boot=2000)
        assert m == pytest.approx(30.0)
        assert lo < m < hi
        # Normal approximation: 1.96 * sqrt(0.3 * 0.7 / 1000) = 2.84 points.
        assert hi - m == pytest.approx(2.84, abs=0.5)
        assert m - lo == pytest.approx(2.84, abs=0.5)

    def test_constant_vector_has_zero_width(self) -> None:
        assert ci.bootstrap_ci(np.ones(50), n_boot=200) == (100.0, 100.0, 100.0)


class Test_paired_diff_ci:
    def test_pairing_narrows_the_interval(self) -> None:
        rng = np.random.default_rng(1)
        b = rng.random(1000) < 0.4
        a = b.copy()
        a[:30] = True  # a is b plus up to 30 extra correct answers
        d, lo, hi = ci.paired_diff_ci(a, b, n_boot=2000)
        assert d == pytest.approx(100 * (a.mean() - b.mean()))
        assert lo > 0  # clearly better although both accuracies are ~40%
        assert hi - lo < 3.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            ci.paired_diff_ci(np.ones(3), np.ones(4))


class Test_compute:
    def test_reads_runs_and_applies_the_1m_override(self, tmp_path) -> None:
        def write(run: str, correct) -> None:
            d = tmp_path / run / "real"
            os.makedirs(d, exist_ok=True)
            with open(d / "pred_real_gsm8k-test.jsonl", "w") as f:
                for c in correct:
                    f.write(json.dumps({"correct": bool(c)}) + "\n")

        write("rft_q1m", [1, 0, 0, 0])
        write("grpo2_q1m_lr2e-5", [1, 1, 0, 0])
        write("grpo2_q1m", [0, 0, 0, 0])  # discarded run: must not be used
        res = ci.compute(str(tmp_path), n_boot=200)
        acc = res["accuracy"]["GSM8K"]["1M"]
        assert acc["rft"]["acc"] == 25.0 and acc["grpo"]["acc"] == 50.0
        assert res["paired_difference"]["GSM8K"]["1M"]["grpo-rft"]["diff"] == 25.0
        assert "SVAMP" not in res["accuracy"]
        assert "| 1M |" in ci.to_markdown(res)
