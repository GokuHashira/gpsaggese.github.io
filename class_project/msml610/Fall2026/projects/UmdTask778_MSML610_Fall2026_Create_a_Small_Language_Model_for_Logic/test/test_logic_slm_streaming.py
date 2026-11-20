"""
CPU tests for streamed pretraining: mixture blocks, corpus builder, step
batching, compact checkpoints, and pretrain.py with --corpus (single process
and 2-process DDP via torchrun on gloo).

Import as:

import test.test_logic_slm_streaming as ttlsst
"""

import json
import os
import subprocess
import sys

import pytest
import torch

import build_corpus
import logic_slm_mixture as lsmix
import logic_slm_model as lsmod
import logic_slm_train as lstrain
import pretrain

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Small mixture: M1 split into two "generators" with different shares.
_MIX = {
    "a": {"module": "logic_slm_math_m1", "share": 3.0, "train": {"n_ops_range": [1, 2]},
          "eval": {"test_id": {"n_ops_range": [1, 2]}}},
    "b": {"module": "logic_slm_math_m1", "share": 1.0, "train": {"n_ops_range": [3, 4]},
          "eval": {"test_id": {"n_ops_range": [3, 4]}, "test_extrap": {"n_ops_range": [5, 6]}}},
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    mix = out / "mix.json"
    mix.write_text(json.dumps(_MIX))
    build_corpus.main([
        "--out", str(out), "--mixture", str(mix), "--n-eval", "50", "--n-val", "100",
        "--vocab-size", "300", "--tok-sample", "2000", "--train-tokens", "20000",
        "--seq-len", "128", "--workers", "2",
    ])
    return out


class Test_build_corpus:
    def test_files(self, corpus) -> None:
        names = sorted(os.listdir(corpus / "eval"))
        assert names == ["a_test_id.jsonl", "b_test_extrap.jsonl", "b_test_id.jsonl"]
        man = json.loads((corpus / "manifest.json").read_text())
        assert man["train_tokens"] == (20000 // 128) * 128
        assert (corpus / "train.bin").stat().st_size == 2 * man["train_tokens"]
        assert man["vocab_size"] <= 300

    def test_heldout_disjoint(self, corpus) -> None:
        held = lsmix.load_held_prompts(str(corpus))
        val = [json.loads(l)["prompt"] for l in open(corpus / "val.jsonl")]
        evals = [json.loads(l)["prompt"] for f in os.listdir(corpus / "eval")
                 for l in open(corpus / "eval" / f)]
        assert not set(val) & set(evals)
        assert len(held) == len(set(val) | set(evals))


class Test_mixture_blocks:
    def test_deterministic_and_shares(self, corpus) -> None:
        held = lsmix.load_held_prompts(str(corpus))
        ds1 = lsmix.MixtureBlocks(_MIX, str(corpus / "tokenizer"), 128, 7, held_prompts=held)
        ds2 = lsmix.MixtureBlocks(_MIX, str(corpus / "tokenizer"), 128, 7, held_prompts=held)
        for j in (0, 5, 123456789):
            assert torch.equal(ds1[j], ds2[j])
            assert ds1[j].shape == (129,)
        assert not torch.equal(ds1[0], ds1[1])
        other = lsmix.MixtureBlocks(_MIX, str(corpus / "tokenizer"), 128, 8, held_prompts=held)
        assert not torch.equal(ds1[0], other[0])

    def test_no_heldout_prompts(self, corpus) -> None:
        held = lsmix.load_held_prompts(str(corpus))
        ds = lsmix.MixtureBlocks(_MIX, str(corpus / "tokenizer"), 128, 0, held_prompts=held)
        text = "".join(
            lsmix.lstok.load_tokenizer(str(corpus / "tokenizer")).decode(ds[j].tolist())
            for j in range(60)
        )
        for p in list(held)[:200]:
            assert p not in text

    def test_stable_seed(self) -> None:
        assert lsmix.stable_seed(1, "x", 2) == lsmix.stable_seed(1, "x", 2)
        assert lsmix.stable_seed(1, "x", 2) != lsmix.stable_seed(1, "x", 3)


class Test_step_batches:
    @pytest.mark.parametrize("world", [1, 2, 4])
    def test_cover_same_blocks_any_world(self, world) -> None:
        seqs, micro = 16, 2
        got = []
        for r in range(world):
            for idx in pretrain.StepBatches(3, 5, seqs, micro, r, world):
                got.extend(idx)
        assert sorted(got) == list(range(3 * seqs, 5 * seqs))


class Test_checkpoints:
    @pytest.mark.parametrize("weights_only", [False, True])
    def test_round_trip(self, tmp_path, weights_only) -> None:
        cfg = lsmod.make_config("tiny", 64, eos_token_id=0, pad_token_id=1)
        m1 = lsmod.make_model(cfg)
        opt = lstrain.make_optimizer(m1, 1e-3, 0.1)
        m1(input_ids=torch.zeros(1, 4, dtype=torch.long)).logits.sum().backward()
        opt.step()
        path = str(tmp_path / "ck.pt")
        lstrain.save_checkpoint(path, m1, opt, 17, weights_only, train_time_s=12.5)
        m2 = lsmod.make_model(cfg)
        opt2 = lstrain.make_optimizer(m2, 1e-3, 0.1)
        assert lstrain.load_checkpoint(path, m2, opt2, "cpu") == (17, 12.5)
        for (k, v1), v2 in zip(m1.state_dict().items(), m2.state_dict().values()):
            tol = 1e-2 if weights_only else 0
            assert torch.allclose(v1, v2, atol=tol), k
        assert bool(opt2.state) == (not weights_only)

    def test_old_checkpoint_without_train_time(self, tmp_path) -> None:
        """
        Checkpoints written before `train_time_s` existed load with 0 seconds.
        """
        cfg = lsmod.make_config("tiny", 64, eos_token_id=0, pad_token_id=1)
        m = lsmod.make_model(cfg)
        path = str(tmp_path / "old.pt")
        torch.save({"step": 5, "model": m.state_dict()}, path)
        opt = lstrain.make_optimizer(m, 1e-3, 0.1)
        assert lstrain.load_checkpoint(path, m, opt, "cpu") == (5, 0.0)

    def test_su_rate(self) -> None:
        assert lstrain.su_rate(99.0) == 99.0
        assert lstrain.su_rate() in (0.0, 7.0, 48.0, 144.0)


class Test_bf16_state_dict:
    def test_model_untouched_and_tied(self) -> None:
        """
        The copy is bf16, the model stays fp32, and tied weights stay shared.
        """
        cfg = lsmod.make_config("tiny", 64, eos_token_id=0, pad_token_id=1)
        m = lsmod.make_model(cfg)
        before = {k: v.clone() for k, v in m.state_dict().items()}
        sd = lstrain.bf16_state_dict(m)
        assert all(v.dtype == torch.bfloat16 for v in sd.values())
        for k, v in m.state_dict().items():
            assert v.dtype == torch.float32 and torch.equal(v, before[k]), k
        assert sd["lm_head.weight"] is sd["model.embed_tokens.weight"]


class Test_setup_dist:
    def test_long_timeout(self, monkeypatch) -> None:
        """
        Multi-process runs use `DIST_TIMEOUT`, not the 10-minute default.
        """
        calls = []
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(lstrain.dist, "init_process_group",
                            lambda backend, **kw: calls.append((backend, kw)))
        monkeypatch.setattr(lstrain.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(lstrain.torch.cuda, "is_available", lambda: False)
        d = lstrain.setup_dist()
        assert d.world == 2
        assert calls == [("gloo", {"timeout": lstrain.DIST_TIMEOUT})]
        assert lstrain.DIST_TIMEOUT.total_seconds() >= 3600


_PT_ARGS = [
    "--size", "tiny", "--tokens", "16384", "--batch-tokens", "4096", "--micro-batch", "4",
    "--seq-len", "128", "--log-every", "2", "--val-every", "2", "--val-blocks", "4",
    "--save-every", "2", "--eval-limit", "4", "--max-new-tokens", "16", "--data-workers", "2",
]


class Test_pretrain_streaming:
    def test_single_process_and_resume(self, corpus, tmp_path) -> None:
        out = tmp_path / "run"
        m = pretrain.main(["--corpus", str(corpus), "--out", str(out), *_PT_ARGS])
        assert (out / "final" / "config.json").exists()
        assert {"a_test_id/exact_match", "b_test_extrap/exact_match"} <= set(m)
        assert not (out / "last.pt").exists()
        cfg = json.loads((out / "final" / "config.json").read_text())
        assert cfg.get("dtype", cfg.get("torch_dtype")) == "bfloat16"

    def test_ddp_two_processes(self, corpus, tmp_path) -> None:
        out = tmp_path / "ddp"
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
               "--master_port=29531", os.path.join(_PROJ, "pretrain.py"),
               "--corpus", str(corpus), "--out", str(out), "--ckpt-weights-only", *_PT_ARGS]
        env = dict(os.environ, PYTHONPATH=_PROJ + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   OMP_NUM_THREADS="1")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=_PROJ, timeout=600)
        assert r.returncode == 0, r.stderr[-3000:]
        metrics = json.loads((out / "metrics.json").read_text())
        assert metrics["gpus"] == 2
        assert (out / "final" / "config.json").exists()


class Test_full_mixture:
    def test_every_generator_builds(self, tmp_path) -> None:
        """
        The default six-generator mixture: eval sets, tokenizer and blocks.
        """
        out = tmp_path / "full"
        build_corpus.main([
            "--out", str(out), "--n-eval", "20", "--n-val", "20", "--vocab-size", "1024",
            "--tok-sample", "300", "--seq-len", "256",
        ])
        evals = sorted(os.listdir(out / "eval"))
        assert len(evals) == sum(len(s["eval"]) for s in lsmix.MIXTURE.values())
        ds = lsmix.MixtureBlocks(lsmix.MIXTURE, str(out / "tokenizer"), 256, 0,
                                 held_prompts=lsmix.load_held_prompts(str(out)))
        tok = lsmix.lstok.load_tokenizer(str(out / "tokenizer"))
        text = "".join(tok.decode(ds[j].tolist()) for j in range(40))
        for marker in ("Problem:", "Premises:", "forall", "Is "):
            assert marker in text, marker


class Test_interrupt_and_resume:
    @pytest.mark.parametrize("nproc", [1, 2])
    def test_deadline_resume_eval_only(self, corpus, tmp_path, nproc) -> None:
        out = tmp_path / f"run{nproc}"
        args = ["--corpus", str(corpus), "--out", str(out), "--ckpt-weights-only",
                "--wandb-project", "", *_PT_ARGS]

        def run(extra):
            if nproc == 1:
                return pretrain.main(args + extra)
            cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
                   f"--master_port={29600 + len(extra)}", os.path.join(_PROJ, "pretrain.py"),
                   *args, *extra]
            env = dict(os.environ, OMP_NUM_THREADS="1",
                       PYTHONPATH=_PROJ + os.pathsep + os.environ.get("PYTHONPATH", ""))
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=_PROJ, timeout=600)
            assert r.returncode == 0, r.stderr[-3000:]
            return None

        # 1) Deadline in the past: stops after the first step with a checkpoint.
        m = run(["--deadline-unix", "1"])
        if nproc == 1:
            assert m == {"stopped_at_step": 1}
        assert (out / "INCOMPLETE").exists() and (out / "last.pt").exists()
        assert not (out / "final").exists()
        ck = torch.load(out / "last.pt")
        assert ck["step"] == 1
        first_job_time = ck["train_time_s"]
        assert first_job_time > 0
        # 2) Resume to completion: marker and checkpoint removed, final saved.
        run([])
        assert (out / "final" / "config.json").exists()
        assert not (out / "INCOMPLETE").exists() and not (out / "last.pt").exists()
        metrics = json.loads((out / "metrics.json").read_text())
        assert "a_test_id/exact_match" in metrics and "train_time_s" in metrics
        # The SU cost includes the time of the first job.
        assert metrics["train_time_s"] > first_job_time
        # 3) Rerun: evaluation only, training stats kept.
        mtime = (out / "final" / "model.safetensors").stat().st_mtime
        run([])
        assert (out / "final" / "model.safetensors").stat().st_mtime == mtime
        again = json.loads((out / "metrics.json").read_text())
        assert again["train_time_s"] == metrics["train_time_s"]

    def test_rewarm_lr_logged(self, corpus, tmp_path, caplog) -> None:
        out = tmp_path / "rw"
        args = ["--corpus", str(corpus), "--out", str(out), "--ckpt-weights-only", *_PT_ARGS]
        pretrain.main(args + ["--deadline-unix", "1"])
        with caplog.at_level("INFO", logger="pretrain"):
            pretrain.main(args + ["--rewarm-steps", "2"])
        assert any("re-warming LR" in r.message for r in caplog.records)
