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
        assert m["best_epoch"] in (1, 2)
        assert {"a_test_id/exact_match", "l_test_id/exact_match"} <= set(m)
