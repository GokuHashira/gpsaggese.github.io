#!/usr/bin/env python
"""
Build corpus v2 (plan v2): real open math data + our synthetic generators.

Steps (all deterministic given --seed):
1. Evaluation sets: synthetic test sets (`<gen>_<split>.jsonl`) plus the real
   benchmarks from `data/processed/real/eval` copied as `real_<name>.jsonl`.
   Synthetic validation examples -> `val.jsonl`.
2. Tokenizer: 32k byte-level BPE (one token per digit) trained on a balanced
   sample of every source.
3. Real sources -> `sources/<name>/tokens.bin` (uint16, each document ends with
   EOS) + `offsets.npy` (document starts, length n_docs + 1). Every document
   with a 13-gram overlap with any real eval question is dropped.
4. Validation tokens `val.bin`: synthetic validation examples + the first
   `holdout` documents of every real source (never sampled for training).
5. `manifest.json`: mixture (`logic_slm_mixture.MIXTURE_V2`), per-source
   document / token / decontamination counts, and the projected number of
   passes over each source for each planned model size.

Example:

> python build_corpus_v2.py --out data/processed/corpus_v2 --workers 32
"""

import argparse
import glob
import gzip
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

import build_corpus as bc
import logic_slm_decontam as lsdec
import logic_slm_format as lsform
import logic_slm_mixture as lsmix
import logic_slm_tokenizer as lstok

_LOG = logging.getLogger("build_corpus_v2")

# name -> (reader kind, input glob relative to the project dir)
SOURCES: Dict[str, Tuple[str, str]] = {
    "finemath": ("parquet_text", "data/raw/finemath4/finemath-4plus/*.parquet"),
    "tinygsm": ("jsonl", "data/processed/real/tinygsm/*.jsonl.gz"),
    "openmathinstruct2": ("jsonl", "data/processed/real/openmathinstruct2/*.jsonl.gz"),
    "metamathqa": ("jsonl", "data/processed/real/metamathqa/*.jsonl.gz"),
    "orcamath": ("jsonl", "data/processed/real/orcamath/*.jsonl.gz"),
    "gsm8k_train": ("jsonl", "data/processed/real/gsm8k-train/*.jsonl.gz"),
}
REAL_EVAL_DIR = "data/processed/real/eval"
# Planned pretraining tokens per model size (plan v2).
PLANNED_TOKENS = {"q125m": 5e9, "q0.5b": 8e9, "q1.5b": 12e9}
TOKEN_DTYPE = np.uint16


def iter_texts(kind: str, path: str, batch: int = 1024) -> Iterator[List[str]]:
    """
    Yield batches of training texts from one input file.
    """
    if kind == "parquet_text":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for rb in pf.iter_batches(batch_size=batch, columns=["text"]):
            yield [t for t in rb.column(0).to_pylist() if t]
    elif kind == "jsonl":
        out: List[str] = []
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                out.append(r["prompt"] + r["solution"])
                if len(out) == batch:
                    yield out
                    out = []
        if out:
            yield out
    else:
        raise ValueError(f"Unknown reader kind {kind}")


# Worker state (set once per process).
_W: Dict[str, object] = {}


def _init_worker(tok_dir: str, index: Set[int]) -> None:
    _W["tok"] = lstok.load_tokenizer(tok_dir)
    _W["index"] = index


def _tokenize_file(job: Tuple[str, str, str]) -> Dict[str, int]:
    """
    Tokenize one input file into `<prefix>.bin` + `<prefix>.len.npy`, dropping
    contaminated documents.
    """
    kind, path, prefix = job
    tok, index = _W["tok"], _W["index"]
    eos = tok.eos_token_id
    lens: List[int] = []
    n_in = n_cont = 0
    with open(prefix + ".bin", "wb") as f:
        for texts in iter_texts(kind, path):
            n_in += len(texts)
            keep = [t for t in texts if not lsdec.is_contaminated(t, index)]
            n_cont += len(texts) - len(keep)
            if not keep:
                continue
            for ids in tok(keep, add_special_tokens=False)["input_ids"]:
                arr = np.asarray(ids + [eos], dtype=TOKEN_DTYPE)
                arr.tofile(f)
                lens.append(arr.size)
    np.save(prefix + ".len.npy", np.asarray(lens, dtype=np.int64))
    return {"docs_in": n_in, "contaminated": n_cont, "docs": len(lens), "tokens": int(sum(lens))}


def _merge(name: str, prefixes: List[str], out_dir: str) -> None:
    """
    Concatenate per-file parts (in input order) into tokens.bin + offsets.npy.
    """
    os.makedirs(out_dir, exist_ok=True)
    lens = [np.load(p + ".len.npy") for p in prefixes]
    offsets = np.concatenate([[0], np.cumsum(np.concatenate(lens))]).astype(np.int64)
    with open(os.path.join(out_dir, "tokens.bin"), "wb") as fout:
        for p in prefixes:
            with open(p + ".bin", "rb") as fin:
                shutil.copyfileobj(fin, fout, length=1 << 26)
            os.remove(p + ".bin")
            os.remove(p + ".len.npy")
    np.save(os.path.join(out_dir, "offsets.npy"), offsets)
    _LOG.info("%s: %d docs, %d tokens", name, len(offsets) - 1, int(offsets[-1]))


def tokenizer_sample(seed: int, n_per_source: int, n_synth: int, max_chars: int) -> List[str]:
    """
    Balanced text sample for tokenizer training.
    """
    texts: List[str] = []
    for name, (kind, pattern) in SOURCES.items():
        got = 0
        for path in sorted(glob.glob(pattern))[:4]:
            for batch in iter_texts(kind, path):
                take = batch[: n_per_source - got]
                texts.extend(t[:max_chars] for t in take)
                got += len(take)
                if got >= n_per_source:
                    break
            if got >= n_per_source:
                break
        _LOG.info("tokenizer sample: %s %d texts", name, got)
    for name in sorted(lsmix.MIXTURE):
        texts.extend(
            lsform.render_example(e)
            for e in lsmix.generate(lsmix.MIXTURE, name, n_synth, lsmix.stable_seed(seed, name, "tok"))
        )
    return texts


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--n-eval", type=int, default=2000, help="synthetic test problems per split")
    ap.add_argument("--n-val", type=int, default=500, help="synthetic validation problems per generator")
    ap.add_argument("--tok-per-source", type=int, default=60_000)
    ap.add_argument("--tok-synth", type=int, default=10_000, help="per generator")
    ap.add_argument("--tok-max-chars", type=int, default=6_000)
    ap.add_argument("--sources", nargs="*", default=list(SOURCES), help="subset of real sources")
    ap.add_argument("--skip-tokenizer", action="store_true", help="reuse <out>/tokenizer")
    a = ap.parse_args(argv)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(a.out, exist_ok=True)

    # 1. Eval and synthetic validation sets.
    synth = {k: v for k, v in lsmix.MIXTURE_V2.items() if not lsmix.is_stored(v)}
    bc.build_eval_and_val(synth, a.out, a.seed, a.n_eval, a.n_val)
    real_eval = sorted(glob.glob(os.path.join(REAL_EVAL_DIR, "*.jsonl")))
    if not real_eval:
        raise FileNotFoundError(f"No real eval sets in {REAL_EVAL_DIR}: run realdata build-eval first")
    for p in real_eval:
        shutil.copy(p, os.path.join(a.out, "eval", "real_" + os.path.basename(p)))
    index, counts = lsdec.load_eval_index(REAL_EVAL_DIR)
    _LOG.info("decontamination index: %d 13-grams from %s", len(index), counts)

    # 2. Tokenizer.
    tok_dir = os.path.join(a.out, "tokenizer")
    if not a.skip_tokenizer:
        tok = lstok.build_tokenizer(
            tokenizer_sample(a.seed, a.tok_per_source, a.tok_synth, a.tok_max_chars), a.vocab_size)
        tok.save_pretrained(tok_dir)
    tok = lstok.load_tokenizer(tok_dir)
    if len(tok) > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError("Vocabulary too large for uint16 tokens")
    _LOG.info("tokenizer: %d tokens", len(tok))

    # 3. Real sources (parallel over input files).
    parts_dir = os.path.join(a.out, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    jobs, by_source = [], {}
    for name in a.sources:
        kind, pattern = SOURCES[name]
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"{name}: no input files match {pattern}")
        by_source[name] = []
        for i, path in enumerate(paths):
            prefix = os.path.join(parts_dir, f"{name}_{i:04d}")
            jobs.append((kind, path, prefix))
            by_source[name].append(prefix)
    stats: Dict[str, Dict[str, int]] = {n: {"files": len(p)} for n, p in by_source.items()}
    with mp.get_context("fork").Pool(a.workers, _init_worker, (tok_dir, index)) as pool:
        for job, st in zip(jobs, pool.imap(_tokenize_file, jobs)):
            name = next(n for n, ps in by_source.items() if job[2] in ps)
            for k, v in st.items():
                stats[name][k] = stats[name].get(k, 0) + v
            _LOG.info("tokenized %s: %s", os.path.basename(job[1]), st)
    for name, prefixes in by_source.items():
        _merge(name, prefixes, os.path.join(a.out, "sources", name))
    os.rmdir(parts_dir)

    # 4. Validation tokens: synthetic val + held-out real documents.
    eos = tok.eos_token_id
    with open(os.path.join(a.out, "val.bin"), "wb") as f:
        for r in map(json.loads, open(os.path.join(a.out, "val.jsonl"))):
            np.asarray(tok.encode(r["prompt"] + r["solution"], add_special_tokens=False) + [eos],
                       dtype=TOKEN_DTYPE).tofile(f)
        for name in sorted(lsmix.MIXTURE_V2):
            spec = lsmix.MIXTURE_V2[name]
            if not lsmix.is_stored(spec) or name not in by_source:
                continue
            src = lsmix.StoredSource(os.path.join(a.out, spec["stored"]), spec.get("holdout", 0))
            for i in range(src.holdout):
                np.asarray(src.doc(i), dtype=TOKEN_DTYPE).tofile(f)

    # 5. Manifest with token counts and projected passes per model size.
    for name in by_source:
        offs = np.load(os.path.join(a.out, "sources", name, "offsets.npy"), mmap_mode="r")
        hold = lsmix.MIXTURE_V2[name].get("holdout", 0)
        stats[name]["train_tokens"] = int(offs[-1] - offs[min(hold, len(offs) - 1)])
        stats[name]["passes"] = {
            size: round(lsmix.MIXTURE_V2[name]["share"] * toks / max(stats[name]["train_tokens"], 1), 2)
            for size, toks in PLANNED_TOKENS.items()
        }
    manifest = {
        "version": "v2", "seed": a.seed, "mixture": lsmix.MIXTURE_V2, "vocab_size": len(tok),
        "format": lsform.FORMAT_VERSION, "sources": stats, "planned_tokens": PLANNED_TOKENS,
        "real_eval_counts": counts,
        "val_tokens": int(os.path.getsize(os.path.join(a.out, "val.bin")) // 2),
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    _LOG.info("done: %s", json.dumps({n: {k: v for k, v in s.items() if k != "files"} for n, s in stats.items()}))


if __name__ == "__main__":
    main()
