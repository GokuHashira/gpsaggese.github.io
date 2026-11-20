"""
Unit tests for the tokenizer, model configs, data files and a CPU smoke run
of `pretrain.py`.

Import as:

import test.test_logic_slm_pipeline as ttlspi
"""

import json
import math

import numpy as np
import pytest
import torch

import logic_slm_data as lsdata
import logic_slm_eval as lseval
import logic_slm_format as lsform
import logic_slm_math_m1 as lsmam1
import logic_slm_model as lsmod
import logic_slm_tokenizer as lstok
import pretrain


def _texts(n=3000, seed=0):
    return [lsform.render_example(e) for e in lsmam1.generate_dataset(n, seed, n_ops_range=(1, 6))]


@pytest.fixture(scope="module")
def tok():
    logic = "Premises: p -> q. q -> r. p.\nQuestion: r?\nSolution:\np, p -> q => q\n#### yes"
    return lstok.build_tokenizer(_texts() + [logic] * 50, vocab_size=512)


def _write_jsonl(path, exs):
    with open(path, "w") as f:
        for e in exs:
            f.write(json.dumps(lsmam1.example_to_record(e)) + "\n")


class Test_tokenizer:
    def test_round_trip(self, tok) -> None:
        texts = _texts(200, seed=5) + ["Premises: ~p | q. forall x (A(x) -> B(x)).\nQuestion: q?"]
        for t in texts:
            assert tok.decode(tok.encode(t, add_special_tokens=False)) == t

    def test_digits_are_single_tokens(self, tok) -> None:
        for t in _texts(200, seed=6):
            for piece in tok.convert_ids_to_tokens(tok.encode(t, add_special_tokens=False)):
                digits = [ch for ch in piece if ch.isdigit()]
                assert len(digits) <= 1 and (not digits or piece in "0123456789"), piece

    def test_reserved_tokens_atomic(self, tok) -> None:
        for sym in lstok.RESERVED_TOKENS:
            ids = tok.encode(f"a {sym} b", add_special_tokens=False)
            assert tok.convert_tokens_to_ids(sym) in ids, sym

    def test_specials(self, tok) -> None:
        assert tok.eos_token == lstok.EOS and tok.pad_token == lstok.PAD
        assert len(tok) <= 512
        ids = lstok.encode_example(tok, "#### 5")
        assert ids[-1] == tok.eos_token_id


class Test_model:
    @pytest.mark.parametrize("size, target", [("25m", 25e6), ("125m", 127e6), ("1b", 1.0e9)])
    def test_sizes(self, size, target) -> None:
        n = lsmod.analytic_non_embedding_params(size)
        assert abs(n - target) / target < 0.05, n

    def test_count_matches_analytic(self) -> None:
        cfg = lsmod.make_config("tiny", 300, eos_token_id=0, pad_token_id=1)
        model = lsmod.make_model(cfg)
        p = lsmod.count_params(model)
        assert p["non_embedding"] == lsmod.analytic_non_embedding_params("tiny")
        assert model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()


class Test_data_and_pretrain:
    def test_bin_and_smoke_run(self, tok, tmp_path) -> None:
        tok_dir = tmp_path / "tok"
        tok.save_pretrained(tok_dir)
        train_j, val_j, test_j = (tmp_path / f"{n}.jsonl" for n in ("train", "val", "test"))
        _write_jsonl(train_j, lsmam1.generate_dataset(3000, 1))
        _write_jsonl(val_j, lsmam1.generate_dataset(300, 2))
        _write_jsonl(test_j, lsmam1.generate_dataset(20, 3))
        n = lsdata.tokenize_jsonl_to_bin([str(train_j)], str(tok_dir), str(tmp_path / "train.bin"))
        lsdata.tokenize_jsonl_to_bin([str(val_j)], str(tok_dir), str(tmp_path / "val.bin"))
        arr = lsdata.load_bin(str(tmp_path / "train.bin"))
        assert arr.size == n and int((arr == tok.eos_token_id).sum()) == 3000
        # The first example decodes back to its text.
        first_eos = int(np.argmax(arr == tok.eos_token_id))
        rec = lsdata.read_records(str(train_j), 1)[0]
        assert tok.decode(arr[:first_eos].tolist()) == rec["prompt"] + rec["solution"]

        out = tmp_path / "run"
        metrics = pretrain.main([
            "--size", "tiny", "--tokenizer", str(tok_dir),
            "--train-bin", str(tmp_path / "train.bin"), "--val-bin", str(tmp_path / "val.bin"),
            "--out", str(out), "--tokens", "65536", "--batch-tokens", "4096",
            "--micro-batch", "8", "--seq-len", "128", "--log-every", "4", "--val-every", "8",
            "--val-blocks", "16", "--eval-jsonl", f"m1={test_j}", "--eval-limit", "8",
            "--max-new-tokens", "24",
        ])
        assert (out / "final" / "config.json").exists()
        assert (out / "pred_m1.jsonl").exists()
        # Better than uniform guessing over the vocabulary.
        assert np.isfinite(metrics["val_loss"]) and metrics["val_loss"] < math.log(len(tok))
        assert 0.0 <= metrics["m1/exact_match"] <= 1.0


class Test_block_stream:
    def test_deterministic_and_resumable(self) -> None:
        toks = np.arange(10_000, dtype=np.uint16)
        s1, s2 = pretrain.BlockStream(toks, 64, 3), pretrain.BlockStream(toks, 64, 3)
        a = s1.batch(0, 400)
        b = torch.cat([s2.batch(0, 150), s2.batch(150, 250)])
        assert torch.equal(a, b)
        assert torch.equal(a[:, 1:], a[:, :-1] + 1)  # contiguous windows
        # One epoch covers every block exactly once.
        starts = sorted(int(x) for x in s1.batch(0, s1.n_blocks)[:, 0])
        assert starts == [i * 64 for i in range(s1.n_blocks)]

    def test_lr_schedule(self) -> None:
        lrs = [pretrain.lr_at(s, 100, 10, 1.0) for s in range(100)]
        assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
        assert lrs[-1] == pytest.approx(0.1, abs=0.01)
        assert all(x >= y for x, y in zip(lrs[10:], lrs[11:]))


class Test_evaluate_jsonl:
    def test_empty_file(self, tmp_path) -> None:
        """
        An empty eval file gives empty metrics instead of crashing.
        """
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert lseval.evaluate_jsonl(None, None, str(path)) == {"n": 0.0}
