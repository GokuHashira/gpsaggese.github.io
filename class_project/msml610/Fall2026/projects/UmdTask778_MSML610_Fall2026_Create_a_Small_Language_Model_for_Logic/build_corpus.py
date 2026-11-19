#!/usr/bin/env python
"""
Build the stored parts of a corpus: evaluation sets, validation set and
tokenizer (and optionally a fixed training token file).

Pretraining normally streams training blocks on the fly
(`logic_slm_mixture.MixtureBlocks`), so only small files are written:

- `<out>/eval/<gen>_<split>.jsonl`: test sets per generator (e.g. in-distribution
  and extrapolation),
- `<out>/val.jsonl`, `<out>/val.bin`: validation examples (exclude eval prompts),
- `<out>/tokenizer/`: BPE tokenizer trained on a sample of the training mix,
- `<out>/manifest.json`: mixture, seed and sizes.

With `--train-tokens N > 0` a fixed `<out>/train.bin` is also written by
concatenating streamed blocks (useful for small, fully reproducible runs).

Import as:

import build_corpus as bcorp
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from typing import Dict, List, Set

import numpy as np

import logic_slm_data as lsdata
import logic_slm_format as lsform
import logic_slm_mixture as lsmix
import logic_slm_tokenizer as lstok

_LOG = logging.getLogger("build_corpus")


def _write_jsonl(path: str, recs: List[dict]) -> None:
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def build_eval_and_val(
    mixture: Dict[str, dict], out: str, seed: int, n_eval: int, n_val: int
) -> Set[str]:
    """
    Write eval and validation sets; return all held-out prompts.
    """
    os.makedirs(os.path.join(out, "eval"), exist_ok=True)
    held: Set[str] = set()
    for name in sorted(mixture):
        for split in mixture[name]["eval"]:
            recs = []
            for e in lsmix.generate(mixture, name, n_eval, lsmix.stable_seed(seed, name, split), split):
                r = lsmix.to_record(mixture, name, e)
                if r["prompt"] not in held:
                    recs.append(r)
            held.update(r["prompt"] for r in recs)
            _write_jsonl(os.path.join(out, "eval", f"{name}_{split}.jsonl"), recs)
            _LOG.info("eval %s/%s: %d", name, split, len(recs))
    val: List[dict] = []
    for name in sorted(mixture):
        recs = []
        for e in lsmix.generate(mixture, name, 3 * n_val, lsmix.stable_seed(seed, name, "val")):
            r = lsmix.to_record(mixture, name, e)
            if r["prompt"] not in held:
                recs.append(r)
            if len(recs) == n_val:
                break
        val.extend(recs)
    held.update(r["prompt"] for r in val)
    _write_jsonl(os.path.join(out, "val.jsonl"), val)
    _LOG.info("val: %d", len(val))
    return held


def build_tokenizer(mixture: Dict[str, dict], out: str, seed: int, vocab_size: int, n: int) -> int:
    texts = [
        lsform.render_example(e)
        for name in sorted(mixture)
        for e in lsmix.generate(mixture, name, n, lsmix.stable_seed(seed, name, "tok"))
    ]
    tok = lstok.build_tokenizer(texts, vocab_size)
    tok.save_pretrained(os.path.join(out, "tokenizer"))
    _LOG.info("tokenizer: %d tokens", len(tok))
    return len(tok)


_DS = None


def _init_worker(mixture, tok_dir, seq_len, seed, held) -> None:
    global _DS
    _DS = lsmix.MixtureBlocks(mixture, tok_dir, seq_len, seed, held_prompts=held)


def _block(j: int) -> np.ndarray:
    # Drop the overlap token: consecutive stored blocks are disjoint.
    return _DS[j][:-1].numpy().astype(lsdata.TOKEN_DTYPE)


def build_train_bin(
    mixture: Dict[str, dict], out: str, seed: int, tokens: float, seq_len: int,
    held: Set[str], workers: int,
) -> int:
    n_blocks = int(tokens // seq_len)
    args = (mixture, os.path.join(out, "tokenizer"), seq_len, seed, held)
    with mp.get_context("spawn").Pool(workers, _init_worker, args) as pool, open(
        os.path.join(out, "train.bin"), "wb"
    ) as f:
        for arr in pool.imap(_block, range(n_blocks), chunksize=64):
            arr.tofile(f)
    return n_blocks * seq_len


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixture", default="", help="JSON mixture (default: built-in)")
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=1000, help="per generator")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--tok-sample", type=int, default=20_000, help="per generator")
    ap.add_argument("--train-tokens", type=float, default=0, help="also write train.bin")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(a.out, exist_ok=True)
    mixture = lsmix.load_mixture(a.mixture)
    held = build_eval_and_val(mixture, a.out, a.seed, a.n_eval, a.n_val)
    vocab = build_tokenizer(mixture, a.out, a.seed, a.vocab_size, a.tok_sample)
    n_val = lsdata.tokenize_jsonl_to_bin(
        [os.path.join(a.out, "val.jsonl")], os.path.join(a.out, "tokenizer"),
        os.path.join(a.out, "val.bin"),
    )
    n_train = 0
    if a.train_tokens > 0:
        n_train = build_train_bin(mixture, a.out, a.seed, a.train_tokens, a.seq_len, held, a.workers)
    manifest = {
        "seed": a.seed, "mixture": mixture, "vocab_size": vocab, "val_tokens": n_val,
        "train_tokens": n_train, "format": lsform.FORMAT_VERSION,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    _LOG.info("done: %s", json.dumps({k: v for k, v in manifest.items() if k != "mixture"}))


if __name__ == "__main__":
    main()
