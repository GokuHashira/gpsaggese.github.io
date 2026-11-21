"""
CPU tests for SFT: loss masking, pass@k evaluation and an end-to-end sft.py
run on a tiny pretrained model.

Import as:

import test.test_logic_slm_sft as ttlssf
"""

import json
import os

import pytest
import torch
import transformers

import build_corpus
import logic_slm_mixture as lsmix
import logic_slm_sft as lssft
import logic_slm_tokenizer as lstok
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
    root = tmp_path_factory.mktemp("sft")
    mix = root / "mix.json"
    mix.write_text(json.dumps(_MIX))
    corpus = root / "corpus"
    build_corpus.main(["--out", str(corpus), "--mixture", str(mix), "--n-eval", "20",
                       "--n-val", "20", "--vocab-size", "400", "--tok-sample", "1000"])
    pre = root / "pre"
    pretrain.main(["--size", "tiny", "--corpus", str(corpus), "--out", str(pre),
                   "--tokens", "16384", "--batch-tokens", "4096", "--micro-batch", "8",
                   "--seq-len", "128", "--log-every", "2", "--val-every", "4",
                   "--val-blocks", "4", "--eval-jsonl", "--data-workers", "0"])
    return root, corpus, pre


class Test_dataset:
    def test_masking(self, setup) -> None:
        _, corpus, _ = setup
        tok = lstok.load_tokenizer(str(corpus / "tokenizer"))
        pairs = [("Problem: 1 + 2 = ?\nSolution:\n", "1 + 2 = 3\n#### 3")]
        ds = lssft.SFTDataset(pairs, tok, 1024)
        b = ds.collate([ds[0], ds[0]])
        ids, labels = b["input_ids"][0], b["labels"][0]
        n_prompt = len(tok.encode(pairs[0][0], add_special_tokens=False))
        assert (labels[:n_prompt] == lssft.IGNORE_INDEX).all()
        assert torch.equal(labels[n_prompt:], ids[n_prompt:])
        assert labels[-1] == tok.eos_token_id
        assert tok.decode(labels[n_prompt:-1].tolist()) == pairs[0][1]

    def test_build_excludes_held(self, setup) -> None:
        _, corpus, _ = setup
        held = lsmix.load_held_prompts(str(corpus))
        pairs = lssft.build_sft_examples(_MIX, 300, 1, held)
        assert len(pairs) == 600
        assert not {p for p, _ in pairs} & held
        assert pairs == lssft.build_sft_examples(_MIX, 300, 1, held)


class Test_masked_loss:
    def test_equals_full_ignore_index_loss(self) -> None:
        """
        Scoring only labelled positions gives the same loss as scoring every
        position with ignore_index (the memory-saving form must not change results).
        """

        class _Fake(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = logits

            def forward(self, input_ids, attention_mask):
                return transformers.modeling_outputs.CausalLMOutput(logits=self.logits)

        torch.manual_seed(0)
        logits = torch.randn(3, 7, 11)
        labels = torch.full((3, 7), lssft.IGNORE_INDEX)
        labels[0, 3:6] = torch.tensor([1, 2, 3])
        labels[2, 1:7] = torch.tensor([4, 5, 6, 7, 8, 9])
        batch = {"input_ids": torch.zeros(3, 7, dtype=torch.long), "labels": labels,
                 "attention_mask": torch.ones(3, 7, dtype=torch.long)}
        got = lssft.masked_lm_loss(_Fake(logits), batch)
        ref = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1),
            ignore_index=lssft.IGNORE_INDEX)
        assert torch.allclose(got, ref)


class Test_sft_script:
    def test_end_to_end(self, setup) -> None:
        root, corpus, pre = setup
        out = root / "sft_out"
        m = sft.main(["--init", str(pre / "final"), "--corpus", str(corpus), "--out", str(out),
                      "--n-examples", "256", "--epochs", "2", "--batch-size", "32",
                      "--micro-batch", "16", "--val-per-gen", "4", "--n-samples", "4",
                      "--ks", "1", "4", "--select-k", "4", "--eval-limit", "4",
                      "--max-new-tokens", "24", "--log-every", "2"])
        assert (out / "best" / "config.json").exists() and (out / "last" / "config.json").exists()
        hist = json.loads((out / "metrics.json").read_text())["per_epoch"]
        assert [h["epoch"] for h in hist] == [1, 2]
        assert {"pass@1", "pass@4", "greedy_exact_match"} <= set(hist[0])
        # Per-source validation (one entry per generator).
        assert {"greedy/a", "greedy/l", "pass@4/a", "pass@4/l"} <= set(hist[0])
        assert m["best_epoch"] in (1, 2)
        assert {"a_test_id/exact_match", "l_test_id/exact_match"} <= set(m)

    def test_no_save_tracks_best_epoch(self, setup) -> None:
        """
        With `--no-save` the best epoch is still reported.
        """
        root, corpus, pre = setup
        out = root / "sft_nosave"
        m = sft.main(["--init", str(pre / "final"), "--corpus", str(corpus), "--out", str(out),
                      "--n-examples", "128", "--epochs", "2", "--batch-size", "32",
                      "--micro-batch", "16", "--val-per-gen", "4", "--n-samples", "4",
                      "--ks", "1", "4", "--select-k", "4", "--eval-limit", "4",
                      "--max-new-tokens", "24", "--no-save"])
        assert m["best_epoch"] in (1, 2)
        assert m["best_select_metric"] >= 0
        assert not (out / "best").exists()


class Test_resume:
    def test_resume_continues_history(self, setup) -> None:
        """
        `--start-epoch 2` trains only epoch 2 and keeps the earlier history.
        """
        root, corpus, pre = setup
        out = root / "sft_resume"
        hist = root / "hist.json"
        hist.write_text(json.dumps([{"epoch": 1, "pass@4": -0.5, "greedy_exact_match": 0.0}]))
        m = sft.main(["--init", str(pre / "final"), "--corpus", str(corpus), "--out", str(out),
                      "--n-examples", "128", "--epochs", "2", "--batch-size", "32",
                      "--micro-batch", "16", "--val-per-gen", "4", "--n-samples", "4",
                      "--ks", "1", "4", "--select-k", "4", "--eval-limit", "4",
                      "--max-new-tokens", "24", "--start-epoch", "2",
                      "--resume-history", str(hist), "--rewarm-steps", "2"])
        per_epoch = json.loads((out / "metrics.json").read_text())["per_epoch"]
        assert [h["epoch"] for h in per_epoch] == [1, 2]
        assert m["best_epoch"] == 2
        assert (out / "best" / "config.json").exists() and (out / "last" / "config.json").exists()

    def test_unbeaten_best_is_kept(self, setup) -> None:
        root, corpus, pre = setup
        hist = root / "hist_hi.json"
        hist.write_text(json.dumps([{"epoch": 1, "pass@4": 2.0}]))
        m = sft.main(["--init", str(pre / "final"), "--corpus", str(corpus),
                      "--out", str(root / "sft_resume_hi"), "--n-examples", "128", "--epochs", "2",
                      "--batch-size", "32", "--micro-batch", "16", "--val-per-gen", "4",
                      "--n-samples", "4", "--ks", "1", "4", "--select-k", "4", "--eval-limit", "4",
                      "--max-new-tokens", "24", "--start-epoch", "2",
                      "--resume-history", str(hist), "--no-save"])
        assert m["best_epoch"] == 1 and m["best_select_metric"] == 2.0


class Test_save:
    def test_bf16_save_keeps_training_weights(self, setup) -> None:
        """
        A bf16 save writes bf16 weights and leaves the fp32 model unchanged.
        """
        root, _, pre = setup
        model = transformers.LlamaForCausalLM.from_pretrained(str(pre / "final"),
                                                              dtype=torch.float32)
        # Perturb the weights so they are not exactly representable in bf16.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 1e-4)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        tok = lstok.load_tokenizer(str(pre / "final"))
        path = root / "saved_bf16"
        sft._save(model, tok, str(path), "bf16")
        for k, v in model.state_dict().items():
            assert v.dtype == torch.float32 and torch.equal(v, before[k]), k
        cfg = json.loads((path / "config.json").read_text())
        assert cfg.get("dtype", cfg.get("torch_dtype")) == "bfloat16"
        loaded = transformers.LlamaForCausalLM.from_pretrained(str(path))
        assert next(loaded.parameters()).dtype == torch.bfloat16
        emb = before["model.embed_tokens.weight"].to(torch.bfloat16)
        assert torch.equal(loaded.get_input_embeddings().weight, emb)
        assert loaded.lm_head.weight.data_ptr() == loaded.get_input_embeddings().weight.data_ptr()


class Test_pass_at_k_eval:
    def test_restores_global_rng(self, setup) -> None:
        """
        Seeded evaluation does not reset the global RNG used for training.
        """
        _, corpus, pre = setup
        model = transformers.LlamaForCausalLM.from_pretrained(str(pre / "final"))
        tok = lstok.load_tokenizer(str(pre / "final"))
        recs = [json.loads(line) for line in open(corpus / "val.jsonl")][:2]
        torch.manual_seed(123)
        expected = torch.rand(3)
        torch.manual_seed(123)
        lssft.pass_at_k_eval(model, tok, recs, n_samples=2, ks=[1, 2], max_new_tokens=8)
        assert torch.equal(torch.rand(3), expected)
