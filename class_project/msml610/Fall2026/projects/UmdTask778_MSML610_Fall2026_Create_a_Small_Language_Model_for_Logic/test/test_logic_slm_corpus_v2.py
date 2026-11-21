"""
CPU tests for corpus v2: stored real-data sources, the corpus builder with
13-gram decontamination, mixed blocks, and pretraining on the result.

Import as:

import test.test_logic_slm_corpus_v2 as ttlscv
"""

import gzip
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import build_corpus_v2 as bcv2
import logic_slm_mixture as lsmix
import logic_slm_tokenizer as lstok
import pretrain

_EVAL_Q = ("Natalia sold clips to forty eight of her friends in April and then she sold half "
           "as many clips in May how many clips did Natalia sell altogether")


def _fake_inputs(root):
    """
    Fake FineMath-like parquet (text docs), a fake example source and a fake
    real eval set. One FineMath doc copies 15 words of the eval question.
    """
    fm = root / "fm"
    fm.mkdir()
    docs = [f"Lesson {i}: to add fractions with a common denominator add the numerators "
            f"and keep the denominator, for example {i}/7 + 2/7 = {i + 2}/7." for i in range(60)]
    docs.append("A worked example. " + " ".join(_EVAL_Q.split()[:15]) + " and more text here.")
    docs.append("x " * 3000)  # one very long document (longer than a block)
    for k in range(2):
        pq.write_table(pa.table({"text": docs[k::2]}), str(fm / f"part{k}.parquet"))
    ex = root / "ex"
    ex.mkdir()
    with gzip.open(ex / "part-00000.jsonl.gz", "wt") as f:
        for i in range(40):
            f.write(json.dumps({"prompt": f"Problem: What is {i} + {i}?\nSolution:\n",
                                "solution": f"{i} + {i} = {2 * i}\n#### {2 * i}", "answer": str(2 * i),
                                "task": "math", "gen": "ex"}) + "\n")
    ev = root / "eval"
    ev.mkdir()
    with open(ev / "gsm8k_test.jsonl", "w") as f:
        f.write(json.dumps({"prompt": f"Problem: {_EVAL_Q}?\nSolution:\n", "answer": "72",
                            "task": "math", "gen": "gsm8k_test", "solution": ""}) + "\n")
    return fm, ex, ev


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("cv2")
    fm, ex, ev = _fake_inputs(root)
    mp = pytest.MonkeyPatch()
    mp.setattr(bcv2, "SOURCES", {"fm": ("parquet_text", str(fm / "*.parquet")),
                                 "ex": ("jsonl", str(ex / "*.jsonl.gz"))})
    mp.setattr(bcv2, "REAL_EVAL_DIR", str(ev))
    mp.setattr(lsmix, "MIXTURE_V2", {
        "fm": {"stored": "sources/fm", "share": 0.5, "holdout": 2},
        "ex": {"stored": "sources/ex", "share": 0.3, "holdout": 2},
        "m1": {**lsmix.MIXTURE["m1"], "share": 0.2},
    })
    out = root / "corpus"
    bcv2.main(["--out", str(out), "--workers", "2", "--n-eval", "10", "--n-val", "10",
               "--vocab-size", "700", "--tok-per-source", "100", "--tok-synth", "50"])
    yield out
    mp.undo()


class Test_build:
    def test_layout_and_manifest(self, corpus) -> None:
        man = json.loads((corpus / "manifest.json").read_text())
        assert set(man["sources"]) == {"fm", "ex"}
        assert man["sources"]["fm"]["contaminated"] == 1
        assert man["sources"]["fm"]["docs"] == 61 and man["sources"]["ex"]["docs"] == 40
        assert set(man["sources"]["fm"]["passes"]) == set(bcv2.PLANNED_TOKENS)
        assert (corpus / "eval" / "real_gsm8k_test.jsonl").exists()
        assert (corpus / "eval" / "m1_test_id.jsonl").exists()
        assert not (corpus / "parts").exists()
        assert man["val_tokens"] > 0 and man["vocab_size"] <= 700

    def test_tokens_and_offsets(self, corpus) -> None:
        tok = lstok.load_tokenizer(str(corpus / "tokenizer"))
        src = lsmix.StoredSource(str(corpus / "sources" / "ex"), holdout=2)
        assert src.n_docs == 40
        assert np.all(np.diff(np.asarray(src.offsets)) > 0)
        d = src.doc(3)
        assert int(d[-1]) == tok.eos_token_id
        assert tok.decode(d[:-1].tolist()) == "Problem: What is 3 + 3?\nSolution:\n3 + 3 = 6\n#### 6"
        fm = lsmix.StoredSource(str(corpus / "sources" / "fm"))
        text = "".join(tok.decode(fm.doc(i)[:-1].tolist()) for i in range(fm.n_docs))
        assert "Natalia sold clips" not in text  # contaminated doc removed


class Test_mixture_blocks_v2:
    def _ds(self, corpus, seed=0):
        man = json.loads((corpus / "manifest.json").read_text())
        return lsmix.MixtureBlocks(man["mixture"], str(corpus / "tokenizer"), 256, seed,
                                   held_prompts=lsmix.load_held_prompts(str(corpus)),
                                   corpus_dir=str(corpus))

    def test_deterministic_mixed(self, corpus) -> None:
        a, b = self._ds(corpus), self._ds(corpus)
        tok = lstok.load_tokenizer(str(corpus / "tokenizer"))
        text = ""
        for j in (0, 1, 7, 99999):
            assert torch.equal(a[j], b[j]) and a[j].shape == (257,)
        for j in range(40):
            text += tok.decode(a[j].tolist())
        assert "Lesson" in text and "What is" in text  # both stored sources appear
        assert not torch.equal(a[0], self._ds(corpus, seed=1)[0])

    def test_holdout_never_sampled(self, corpus) -> None:
        ds = self._ds(corpus)
        ds._setup()
        src = ds._stored["ex"]
        held = {tuple(src.doc(i).tolist()) for i in range(src.holdout)}
        rng = __import__("random").Random(0)
        for _ in range(2000):
            i = rng.randrange(src.holdout, src.n_docs)
            assert tuple(src.doc(i).tolist()) not in held


class Test_pretrain_on_v2:
    def test_qtiny_two_steps(self, corpus, tmp_path) -> None:
        out = tmp_path / "pt"
        m = pretrain.main(["--size", "qtiny", "--corpus", str(corpus), "--out", str(out),
                           "--tokens", "8192", "--batch-tokens", "4096", "--micro-batch", "8",
                           "--seq-len", "128", "--val-blocks", "2", "--eval-limit", "2",
                           "--max-new-tokens", "8", "--data-workers", "0",
                           "--ledger", str(tmp_path / "runs.jsonl")])
        assert (out / "final" / "config.json").exists()
        assert any(k.startswith("real_gsm8k_test/") for k in m)
