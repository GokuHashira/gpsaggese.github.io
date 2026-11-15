"""
Tokenize JSONL example files into flat uint16 token files for pretraining.

Each example is encoded as prompt + solution + EOS and the examples are
concatenated; the trainer cuts the stream into fixed-length blocks.

Import as:

import logic_slm_data as lsdata
"""

import argparse
import json
import logging
from typing import Iterator, List

import numpy as np

import logic_slm_tokenizer as lstok

_LOG = logging.getLogger(__name__)
TOKEN_DTYPE = np.uint16


def tokenize_jsonl_to_bin(
    jsonl_paths: List[str], tokenizer_dir: str, out_path: str, batch_size: int = 4096
) -> int:
    """
    Tokenize records and write a flat token file.

    :return: number of tokens written
    """
    tok = lstok.load_tokenizer(tokenizer_dir)
    if len(tok) > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError("Vocabulary too large for uint16 tokens")
    eos = tok.eos_token_id
    n_tokens = 0
    with open(out_path, "wb") as f:
        for batch in _batches(lstok.iter_jsonl_texts(jsonl_paths), batch_size):
            enc = tok(batch, add_special_tokens=False)["input_ids"]
            flat = np.fromiter(
                (t for ids in enc for t in ids + [eos]), dtype=TOKEN_DTYPE
            )
            flat.tofile(f)
            n_tokens += flat.size
    _LOG.info("Wrote %d tokens to %s", n_tokens, out_path)
    return n_tokens


def load_bin(path: str) -> np.ndarray:
    """
    Memory-map a token file.
    """
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def _batches(it: Iterator[str], n: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def read_records(path: str, limit: int = 0) -> List[dict]:
    """
    Read JSONL records (optionally only the first `limit`).
    """
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def _main() -> None:
    ap = argparse.ArgumentParser(description="Tokenize JSONL files to a .bin file.")
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    tokenize_jsonl_to_bin(a.jsonl, a.tokenizer, a.out)


if __name__ == "__main__":
    _main()
