"""
Word-level n-gram decontamination of training data against evaluation sets.

Text is lowercased, punctuation is replaced by spaces and the result is split
on whitespace. Every run of `n` (default 13) consecutive words is hashed with
blake2b (8 bytes), which is stable across processes and runs (unlike
`hash()`, which depends on PYTHONHASHSEED). A training record is
contaminated if any of its n-grams occurs in the index built from the
evaluation prompts. Texts shorter than `n` words contribute their whole word
sequence as one n-gram, so short eval questions are still protected.

CLI:

python logic_slm_decontam.py \
    --in-dir data/processed/real/gsm8k-train \
    --out-dir data/processed/real/gsm8k-train.decontam \
    --eval-dir data/processed/real/eval --workers 16

Import as:

import logic_slm_decontam as lsdeco
"""

import argparse
import collections
import functools
import glob
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import re
import time
from typing import Dict, Iterable, List, Set, Tuple

DEFAULT_N = 13
_PUNCT_RE = re.compile(r"[^\w\s]|_")
# Index shared with worker processes (set once per worker by `_init_worker`).
_INDEX: Set[int] = set()


def tokenize(text: str) -> List[str]:
    """
    Lowercase, replace punctuation by spaces and split on whitespace.
    """
    return _PUNCT_RE.sub(" ", text.lower()).split()


def _hash(words: Iterable[str]) -> int:
    h = hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def ngram_hashes(text: str, n: int = DEFAULT_N) -> Set[int]:
    """
    Stable 64-bit hashes of all word n-grams of `text`.

    A text with fewer than `n` (but at least one) words yields one hash of
    its whole word sequence.
    """
    words = tokenize(text)
    if not words:
        return set()
    if len(words) < n:
        return {_hash(words)}
    return {_hash(words[i : i + n]) for i in range(len(words) - n + 1)}


def build_index(texts: Iterable[str], n: int = DEFAULT_N) -> Set[int]:
    """
    Build the set of hashed n-grams of the (evaluation) texts.
    """
    index: Set[int] = set()
    for t in texts:
        index |= ngram_hashes(t, n)
    return index


def is_contaminated(text: str, index: Set[int], n: int = DEFAULT_N) -> bool:
    """
    Whether any word n-gram of `text` occurs in `index`.

    Short texts (< n words) are only compared as a whole, so they never match
    a long eval text by accident.
    """
    words = tokenize(text)
    if not words:
        return False
    if len(words) < n:
        return _hash(words) in index
    for i in range(len(words) - n + 1):
        if _hash(words[i : i + n]) in index:
            return True
    return False


def strip_prompt(prompt: str) -> str:
    """
    Remove format scaffolding ("Problem:", "Premises:", "Question:",
    "Solution:") so only the problem text is indexed.
    """
    lines = []
    for line in prompt.split("\n"):
        s = line.strip()
        if s == "Solution:":
            continue
        for head in ("Problem:", "Premises:", "Question:"):
            if s.startswith(head):
                s = s[len(head) :].strip()
        if s:
            lines.append(s)
    return "\n".join(lines)


def record_text(rec: dict) -> str:
    """
    Text of a training record that is checked: problem + solution.
    """
    return strip_prompt(rec.get("prompt", "")) + "\n" + rec.get("solution", "")


def load_eval_index(eval_dir: str, n: int = DEFAULT_N) -> Tuple[Set[int], Dict[str, int]]:
    """
    Build the index from the prompts of every `*.jsonl` in `eval_dir`.

    :return: (index, number of eval records per file)
    """
    counts: Dict[str, int] = {}
    texts: List[str] = []
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.jsonl"))):
        with open(path) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        counts[os.path.basename(path)] = len(recs)
        texts.extend(strip_prompt(r["prompt"]) for r in recs)
    if not texts:
        raise FileNotFoundError(f"No eval records in {eval_dir}")
    return build_index(texts, n), counts


def _init_worker(index: Set[int]) -> None:
    global _INDEX
    _INDEX = index


def filter_shard(in_path: str, out_path: str, n: int = DEFAULT_N) -> Dict[str, int]:
    """
    Copy the records of a gzip JSONL shard that are not contaminated.

    Uses the worker-global index set by `_init_worker`.

    :return: counts {"in", "kept", "removed", "removed:<gen>"}
    """
    stats: Dict[str, int] = collections.Counter()
    tmp = out_path + ".tmp"
    with gzip.open(in_path, "rt") as fin, gzip.open(tmp, "wt", compresslevel=6) as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            stats["in"] += 1
            if is_contaminated(record_text(rec), _INDEX, n):
                stats["removed"] += 1
                stats[f"removed:{rec.get('gen', '?')}"] += 1
            else:
                stats["kept"] += 1
                fout.write(line if line.endswith("\n") else line + "\n")
    os.replace(tmp, out_path)
    return dict(stats)


def _filter_job(paths: Tuple[str, str], n: int) -> Tuple[str, Dict[str, int]]:
    return paths[0], filter_shard(paths[0], paths[1], n)


def filter_dir(
    in_dir: str,
    out_dir: str,
    index: Set[int],
    *,
    n: int = DEFAULT_N,
    workers: int = 1,
) -> Dict[str, int]:
    """
    Filter every `*.jsonl.gz` shard of `in_dir` into `out_dir` (same names).

    :return: aggregated counts
    """
    shards = sorted(glob.glob(os.path.join(in_dir, "*.jsonl.gz")))
    if not shards:
        raise FileNotFoundError(f"No *.jsonl.gz shards in {in_dir}")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(p, os.path.join(out_dir, os.path.basename(p))) for p in shards]
    total: Dict[str, int] = collections.Counter()
    fn = functools.partial(_filter_job, n=n)
    if workers <= 1:
        _init_worker(index)
        results = map(fn, jobs)
    else:
        # fork shares the index copy-on-write with the workers.
        pool = mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=(index,))
        results = pool.imap_unordered(fn, jobs)
    for _, st in results:
        total.update(st)
    if workers > 1:
        pool.close()
        pool.join()
    return dict(total)


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", nargs="+", required=True, help="dirs of *.jsonl.gz shards")
    ap.add_argument(
        "--out-dir",
        nargs="+",
        required=True,
        help="output dir per input dir (same order)",
    )
    ap.add_argument("--eval-dir", default="data/processed/real/eval")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--report", default=None, help="JSON report path")
    a = ap.parse_args()
    if len(a.in_dir) != len(a.out_dir):
        ap.error("--in-dir and --out-dir need the same number of entries")
    t0 = time.time()
    index, eval_counts = load_eval_index(a.eval_dir, a.n)
    print(f"eval index: {len(index):,} {a.n}-grams from {eval_counts}", flush=True)
    report = {"n": a.n, "eval": eval_counts, "sources": {}}
    for src, dst in zip(a.in_dir, a.out_dir):
        st = filter_dir(src, dst, index, n=a.n, workers=a.workers)
        report["sources"][src] = st
        print(
            f"{src}: in={st.get('in', 0):,} kept={st.get('kept', 0):,} "
            f"removed={st.get('removed', 0):,} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
