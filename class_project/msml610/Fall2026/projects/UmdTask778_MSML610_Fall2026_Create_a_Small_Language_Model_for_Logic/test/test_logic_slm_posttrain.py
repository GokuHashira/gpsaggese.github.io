"""
CPU tests for post-training data: problem pool (no overlap with SFT data),
scoring, RFT pairs, the sampling script and SFT on RFT pairs.

Import as:

import test.test_logic_slm_posttrain as ttlspt
"""

import gzip
import json

import pytest

import build_corpus
import logic_slm_posttrain as lspost
import logic_slm_sft as lssft
import posttrain_sample
import pretrain
import sft

_MIX = {
    "a": {"module": "logic_slm_math_m1", "share": 1.0, "train": {"n_ops_range": [1, 2]},
          "eval": {"test_id": {"n_ops_range": [1, 2]}}},
    "l": {"module": "logic_slm_logic", "share": 1.0, "train": {"level": "L1", "depth_range": [1, 2]},
          "eval": {"test_id": {"level": "L1", "depth_range": [1, 2]}}},
}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("post")
    mix = root / "mix.json"
    mix.write_text(json.dumps(_MIX))
    corpus = root / "corpus"
    build_corpus.main(["--out", str(corpus), "--mixture", str(mix), "--n-eval", "20",
                       "--n-val", "20", "--vocab-size", "400", "--tok-sample", "1000"])
    pre = root / "pre"
    pretrain.main(["--size", "tiny", "--corpus", str(corpus), "--out", str(pre),
                   "--tokens", "16384", "--batch-tokens", "4096", "--micro-batch", "8",
                   "--seq-len", "128", "--log-every", "2", "--val-every", "4",
                   "--val-blocks", "4", "--eval-jsonl", "--data-workers", "0",
                   "--ledger", str(root / "runs.jsonl")])
    return root, corpus, pre


def _fake_source(dirpath, n):
    dirpath.mkdir(parents=True)
    for shard in range(3):
        with gzip.open(dirpath / f"part{shard}.jsonl.gz", "wt") as f:
            for i in range(n):
                k = shard * n + i
                f.write(json.dumps({"prompt": f"Problem: what is {k} + 1?\nSolution:\n",
                                    "solution": f"{k} + 1 = {k + 1}\n#### {k + 1}",
                                    "answer": str(k + 1), "task": "math"}) + "\n")


class Test_pool:
    def test_real_pool_skips_sft_data(self, tmp_path) -> None:
        _fake_source(tmp_path / "src", 40)
        spec = {"src": (str(tmp_path / "src" / "*.jsonl.gz"), 30)}
        sft_pairs, sft_val, _ = lssft.load_real_pairs(spec, 7, set(), 5, 1.0)
        pool = lspost.real_pool(spec, 7, set(), 5, 1.0, {"src": 50})
        used = {p for p, _ in sft_pairs} | {r["prompt"] for r in sft_val}
        assert len(pool) == 50 and len(used) == 35
        assert not used & {p["prompt"] for p in pool}
        assert all(p["gen"] == "src" and p["task"] == "math" for p in pool)

    def test_synthetic_pool_new_and_not_held(self, setup) -> None:
        _, corpus, _ = setup
        held = lspost.lsmix.load_held_prompts(str(corpus))
        pool = lspost.synthetic_pool(_MIX, 40, 1, held)
        assert len(pool) == 80 and not {p["prompt"] for p in pool} & held
        sft_prompts = {p for p, _ in lssft.build_sft_examples(_MIX, 40, 1, held)}
        assert len({p["prompt"] for p in pool} & sft_prompts) < 0.2 * len(pool)


class Test_scoring_and_rft:
    def test_score_and_pairs(self) -> None:
        probs = [{"prompt": "P1", "answer": "3", "task": "math", "gen": "x"},
                 {"prompt": "P2", "answer": "yes", "task": "logic", "gen": "y"}]
        comps = [["1 + 2 = 3\n#### 3", "1 + 2 = 3\n####  3", "1 + 1 = 2\n#### 2", "no answer"],
                 ["p => q\n#### yes", "#### no", "p => q\n#### yes", "q => p\n#### yes"]]
        samples = lspost.score_samples(probs, comps)
        assert samples[0]["correct"] == [True, True, False, False]
        assert samples[1]["correct"] == [True, False, True, True]
        pairs = lspost.rft_pairs(samples, max_per_problem=5)
        # P1: the two correct completions differ only in whitespace -> one pair.
        assert sorted(p for p, _ in pairs) == ["P1", "P2", "P2"]
        assert all(c.rsplit("####", 1)[1].strip() in ("3", "yes") for _, c in pairs)
        assert len(lspost.rft_pairs(samples, max_per_problem=1)) == 2
        summ = lspost.summarize(samples)
        assert summ["x/pass@1"] == 0.5 and summ["all/n"] == 2


class Test_scripts:
    def test_sample_then_rft(self, setup) -> None:
        root, corpus, pre = setup
        out = root / "samples"
        summ = posttrain_sample.main([
            "--model", str(pre / "final"), "--out", str(out), "--corpus", str(corpus),
            "--per-real", "0", "--proofwriter", "0", "--synth-unit", "8", "--synth-override", "",
            "--n-samples", "2", "--max-new-tokens", "16", "--backend", "hf", "--batch-size", "8",
            "--real-eval-dir", str(root / "none")])
        assert summ["backend"] == "hf" and summ["n_problems"] == 16
        rows = [json.loads(line) for line in open(out / "samples.jsonl")]
        assert len(rows) == 16 and all(len(r["completions"]) == 2 for r in rows)
        # SFT on hand-made RFT pairs (the tiny model rarely gets anything right).
        pairs = out / "pairs.jsonl"
        pairs.write_text("".join(json.dumps({"prompt": r["prompt"], "solution": "x\n#### 1"}) + "\n"
                                 for r in rows))
        m = sft.main(["--init", str(pre / "final"), "--corpus", str(corpus),
                      "--out", str(root / "rft"), "--n-examples", "64", "--extra-pairs", str(pairs),
                      "--epochs", "1", "--batch-size", "16", "--micro-batch", "8",
                      "--val-per-gen", "2", "--n-samples", "2", "--ks", "1", "2", "--select-k", "2",
                      "--eval-limit", "2", "--max-new-tokens", "16", "--no-save",
                      "--ledger", str(root / "runs.jsonl")])
        assert m["best_epoch"] == 1
        ledger = [json.loads(line) for line in open(root / "runs.jsonl")]
        counts = [r for r in ledger if r["kind"] == "sft"][-1]["data_counts"]
        assert counts["extra:samples"] == 16


class Test_grpo:
    def test_rewards(self) -> None:
        import grpo

        comps = ["1 + 2 = 3\n#### 3", "#### 4", "3", "x\n#### 3\n#### 3"]
        assert grpo.correctness_reward(comps, answer=["3"] * 4, task=["math"] * 4) == [1.0, 0.0, 0.0, 1.0]
        assert grpo.format_reward(comps) == [1.0, 1.0, 0.0, 0.0]
        assert grpo.correctness_reward(["p\n#### Yes"], answer=["yes"], task=["logic"]) == [1.0]

    def test_two_steps_on_cpu(self, setup) -> None:
        import grpo

        root, corpus, pre = setup
        probs = root / "grpo_problems.jsonl"
        probs.write_text("".join(json.dumps({"prompt": f"Problem: {i} + 1 = ?\nSolution:\n",
                                             "answer": str(i + 1), "task": "math", "gen": "a"}) + "\n"
                                 for i in range(8)))
        out = root / "grpo"
        m = grpo.main(["--init", str(pre / "final"), "--out", str(out), "--problems", str(probs),
                       "--steps", "2", "--num-generations", "2", "--prompts-per-step", "2",
                       "--micro-batch", "4", "--max-completion-length", "16", "--cpu",
                       "--save-every", "100", "--ledger", str(root / "runs.jsonl")])
        assert (out / "final" / "config.json").exists()
        assert (out / "reward_history.json").exists()
        assert m["steps"] == 2 and m["n_problems"] == 8
        ledger = [json.loads(line) for line in open(root / "runs.jsonl")]
        assert ledger[-1]["kind"] == "grpo"


class Test_reuse:
    def test_reuse_takes_sft_records_and_tags_them(self, tmp_path) -> None:
        _fake_source(tmp_path / "src", 10)
        spec = {"src": (str(tmp_path / "src" / "*.jsonl.gz"), 30)}
        # SFT (5 val + 30 train) would leave nothing unseen in 30 records.
        assert lspost.real_pool(spec, 7, set(), 5, 1.0, {"src": 30}) == []
        pool = lspost.real_pool(spec, 7, set(), 5, 1.0, {"src": 12}, reuse={"src"})
        assert len(pool) == 12 and all(p["seen_in_sft"] for p in pool)
        samples = lspost.score_samples(pool, [["#### 0"]] * len(pool))
        summ = lspost.summarize(samples)
        assert summ["seen_in_sft/n"] == 12 and "unseen/n" not in summ


class Test_pool_skip:
    def test_next_stage_gets_disjoint_problems(self, tmp_path) -> None:
        _fake_source(tmp_path / "src", 40)
        spec = {"src": (str(tmp_path / "src" / "*.jsonl.gz"), 30)}
        rft = lspost.real_pool(spec, 7, set(), 5, 1.0, {"src": 20})
        grpo = lspost.real_pool(spec, 7, set(), 5, 1.0, {"src": 20}, extra_skip=20)
        assert len(rft) == 20 and len(grpo) == 20
        assert not {p["prompt"] for p in rft} & {p["prompt"] for p in grpo}
        # Reused sources ignore the extra skip (all their problems are already seen).
        again = lspost.real_pool(spec, 7, set(), 5, 1.0, {"src": 10}, reuse={"src"}, extra_skip=20)
        assert [p["prompt"] for p in again] == [p["prompt"] for p in lspost.real_pool(
            spec, 7, set(), 5, 1.0, {"src": 10}, reuse={"src"})]

    def test_synthetic_tag_changes_problems(self, setup) -> None:
        _, corpus, _ = setup
        held = lspost.lsmix.load_held_prompts(str(corpus))
        a = {p["prompt"] for p in lspost.synthetic_pool(_MIX, 40, 1, held)}
        b = {p["prompt"] for p in lspost.synthetic_pool(_MIX, 40, 1, held, tag="grpo")}
        assert len(a & b) < 0.3 * len(a)


class Test_length_filter:
    def test_drops_prompts_longer_than_the_budget(self, setup) -> None:
        _, _, pre = setup
        import transformers

        tok = transformers.AutoTokenizer.from_pretrained(str(pre / "final"))
        probs = [{"prompt": "Problem: 1 + 1 = ?\nSolution:\n", "answer": "2", "task": "math", "gen": "a"},
                 {"prompt": "Problem: " + "1 + " * 400 + "1 = ?\nSolution:\n", "answer": "401",
                  "task": "math", "gen": "a"}]
        kept, dropped = lspost.filter_by_length(probs, tok, 100)
        assert dropped == 1 and kept == probs[:1]
        assert lspost.filter_by_length(probs, tok, 100000) == (probs, 0)


class Test_e11_pairs:
    def test_pool_records_carry_gold_solutions(self, tmp_path, setup) -> None:
        _fake_source(tmp_path / "src", 10)
        spec = {"src": (str(tmp_path / "src" / "*.jsonl.gz"), 5)}
        pool = lspost.real_pool(spec, 7, set(), 2, 1.0, {"src": 5})
        assert pool and all(p["solution"].endswith(f"#### {p['answer']}") for p in pool)
        _, corpus, _ = setup
        held = lspost.lsmix.load_held_prompts(str(corpus))
        synth = lspost.synthetic_pool(_MIX, 20, 1, held, tag="grpo")
        assert all("####" in p["solution"] for p in synth)


class Test_grpo_dropout_and_guard:
    def test_mismatch_rule(self) -> None:
        import grpo

        key = grpo.IS_RATIO_KEY
        assert grpo.sampler_mismatch({key: 0.27}, 10, 0.7, 10)
        assert not grpo.sampler_mismatch({key: 0.98}, 10, 0.7, 10)
        assert not grpo.sampler_mismatch({key: 0.27}, 5, 0.7, 10)  # too early to judge
        assert not grpo.sampler_mismatch({"loss": 0.1}, 50, 0.7, 10)  # no vLLM: key not logged

    def test_guard_stops_training(self) -> None:
        import types

        import grpo

        guard = grpo.make_mismatch_guard(0.7, 10)
        control = types.SimpleNamespace(should_training_stop=False)
        guard.on_log(None, types.SimpleNamespace(global_step=10), control, logs={grpo.IS_RATIO_KEY: 0.99})
        assert guard.tripped is None and not control.should_training_stop
        guard.on_log(None, types.SimpleNamespace(global_step=15), control, logs={grpo.IS_RATIO_KEY: 0.27})
        assert guard.tripped == 0.27 and control.should_training_stop

    def test_trains_without_the_checkpoint_dropout(self, setup) -> None:
        """
        SFT / RFT checkpoints carry attention_dropout=0.1; GRPO must run with 0
        (the sampler has no dropout), whatever the checkpoint config says.
        """
        import shutil

        import grpo

        root, corpus, pre = setup
        init = root / "with_dropout"
        shutil.copytree(pre / "final", init)
        cfg = json.loads((init / "config.json").read_text())
        cfg["attention_dropout"] = 0.1
        (init / "config.json").write_text(json.dumps(cfg))
        probs = root / "grpo_problems2.jsonl"
        probs.write_text("".join(json.dumps({"prompt": f"Problem: {i} + 2 = ?\nSolution:\n",
                                             "answer": str(i + 2), "task": "math", "gen": "a"}) + "\n"
                                 for i in range(8)))
        out = root / "grpo_nodrop"
        m = grpo.main(["--init", str(init), "--out", str(out), "--problems", str(probs),
                       "--steps", "2", "--num-generations", "2", "--prompts-per-step", "2",
                       "--micro-batch", "4", "--max-completion-length", "16", "--cpu",
                       "--ledger", str(root / "runs.jsonl")])
        assert m["aborted_is_ratio"] is None and m["steps_done"] == 2
        final_cfg = json.loads((out / "final" / "config.json").read_text())
        assert final_cfg["attention_dropout"] == 0.0
