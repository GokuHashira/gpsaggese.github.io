"""
Tests for `logic_slm_metrics.py` and an end-to-end CPU run of `evaluate.py`.

Import as:

import test.test_logic_slm_metrics as ttlsme
"""

import json
import multiprocessing as mp
import os
import time

import pytest

import build_corpus
import evaluate
import logic_slm_metrics as lsmet
import pretrain


def _append_many(path: str, k: int) -> None:
    led = lsmet.RunLedger(path)
    for i in range(50):
        led.append({"writer": k, "i": i, "pad": "x" * 2000})


class Test_accounting:
    def test_flops_and_mfu(self) -> None:
        fpt = lsmet.flops_per_token(100, n_layers=2, hidden=8, seq_len=16, vocab=10)
        assert fpt == 6 * 100 + 12 * 2 * 8 * 16 + 6 * 8 * 10
        assert lsmet.peak_flops("NVIDIA H100 80GB HBM3") == 989e12
        assert lsmet.peak_flops("NVIDIA A100-SXM4-40GB MIG 1g.5gb") == pytest.approx(312e12 / 7)
        assert lsmet.peak_flops("cpu") == 0.0
        assert lsmet.mfu(1e6, 1e9, 1, "NVIDIA H100") == pytest.approx(1e15 / 989e12)
        assert lsmet.mfu(1e6, 1e9, 1, "cpu") == 0.0

    def test_cost_fields(self) -> None:
        c = lsmet.cost_fields(7200, {"train": 3600, "eval": 1800}, 4, 144)
        assert c["gpu_hours"] == pytest.approx(8.0)
        assert c["su"] == pytest.approx(1152.0)
        assert c["su_train"] == pytest.approx(576.0) and c["su_eval"] == pytest.approx(288.0)

    def test_su_per_correct_uses_real_cost_keys(self) -> None:
        # Regression: the summary looked up a key cost_fields() never produces.
        cost = lsmet.cost_fields(3600, {"load": 60, "eval": 1800}, 1, 144)
        assert evaluate.su_per_correct(cost, 100) == pytest.approx(72.0 / 100)
        assert evaluate.su_per_correct(cost, 0) is None

    def test_timer(self) -> None:
        t = lsmet.Timer()
        with t.phase("a"):
            time.sleep(0.05)
        with t.phase("a"):
            time.sleep(0.05)
        assert 0.09 < t.phases["a"] < 0.5 and t.total() >= t.phases["a"]

    def test_gpu_monitor_without_gpu(self, tmp_path) -> None:
        m = lsmet.GpuMonitor(str(tmp_path), interval_s=0.1).start()
        time.sleep(0.3)
        s = m.stop()
        assert set(s) == {"gpu_util_mean_pct", "gpu_mem_peak_mib", "gpu_power_mean_w", "energy_kwh"}

    def test_ledger_concurrent_writers(self, tmp_path) -> None:
        path = str(tmp_path / "runs.jsonl")
        procs = [mp.get_context("spawn").Process(target=_append_many, args=(path, k)) for k in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        recs = lsmet.RunLedger(path).read()  # every line parses: no interleaving
        assert len(recs) == 200 and {r["writer"] for r in recs} == {0, 1, 2, 3}

    def test_git_and_slurm(self) -> None:
        assert lsmet.git_commit(os.path.dirname(os.path.abspath(__file__)))
        assert "host" in lsmet.slurm_info()


_MIX = {"a": {"module": "logic_slm_math_m1", "share": 1.0, "train": {"n_ops_range": [1, 2]},
              "eval": {"test_id": {"n_ops_range": [1, 2]}}}}


class Test_evaluate_end_to_end:
    @pytest.mark.parametrize("size", ["tiny", "qtiny"])
    def test_tiny_model(self, tmp_path, size) -> None:
        mix = tmp_path / "mix.json"
        mix.write_text(json.dumps(_MIX))
        corpus = tmp_path / "corpus"
        build_corpus.main(["--out", str(corpus), "--mixture", str(mix), "--n-eval", "12",
                           "--n-val", "40", "--vocab-size", "300", "--tok-sample", "500"])
        pre = tmp_path / "pre"
        pretrain.main(["--size", size, "--corpus", str(corpus), "--out", str(pre),
                       "--tokens", "8192", "--batch-tokens", "4096", "--micro-batch", "8",
                       "--seq-len", "128", "--val-blocks", "2", "--eval-jsonl", "--data-workers", "0",
                       "--ledger", str(tmp_path / "runs.jsonl")])
        out = tmp_path / "ev"
        ledger = tmp_path / "runs.jsonl"
        res = evaluate.main(["--model", str(pre / "final"), "--name", "tiny", "--stage", "pretrain",
                             "--eval-dir", str(corpus / "eval"), "--sets", "a_*", "nomatch_*",
                             "--out", str(out), "--limit", "6",
                             "--n-samples", "4", "--sample-limit", "3", "--max-new-tokens", "16",
                             "--ledger", str(ledger)])
        r = res["a_test_id"]
        assert {"exact_match", "pass@1", "pass@4", "majority@4", "greedy_tokens_per_s",
                "mean_solution_tokens", "sample_gen_tokens"} <= set(r)
        assert (out / "pred_a_test_id.jsonl").exists() and (out / "eval_metrics.json").exists()
        assert "a_test_id" in json.loads((out / "eval_metrics.partial.json").read_text())
        recs = lsmet.RunLedger(str(ledger)).read()
        kinds = [x["kind"] for x in recs]
        assert kinds == ["pretrain", "eval"]
        assert recs[0]["arch"] == ("qwen2" if size == "qtiny" else "llama")
        ev = recs[-1]
        for k in ("gpu_hours", "su", "wall_time_s", "time_eval_s", "energy_kwh", "git_commit", "params_total"):
            assert k in ev, k
        pt = recs[0]
        for k in ("tokens_trained", "tokens_per_s", "mfu", "flops_per_token", "time_train_s", "su",
                  "val_loss", "params_non_embedding"):
            assert k in pt, k
