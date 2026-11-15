"""
Byte-level BPE tokenizer for math and logic text.

- Every digit is its own token: digits are split before BPE, so merges never
  join digits (key for arithmetic).
- Format markers and logic symbols are reserved tokens that are never split.
- Byte-level, so any text round-trips exactly (no unknown tokens).

Import as:

import logic_slm_tokenizer as lstok
"""

import argparse
import json
import logging
from typing import Iterable, Iterator, List

import tokenizers
import transformers
from tokenizers import decoders, models, pre_tokenizers, trainers

import logic_slm_format as lsform

_LOG = logging.getLogger(__name__)

EOS = "<|endoftext|>"
PAD = "<|pad|>"
SPECIAL_TOKENS = [EOS, PAD]
RESERVED_TOKENS = [
    lsform.ANSWER_MARKER,
    "->",
    "=>",
    "forall",
    "exists",
    "Problem:",
    "Premises:",
    "Question:",
    lsform.SOLUTION_HEADER,
]


def build_tokenizer(
    texts: Iterable[str], vocab_size: int = 2048
) -> transformers.PreTrainedTokenizerFast:
    """
    Train the tokenizer on `texts`.

    :param texts: training texts (rendered examples)
    :param vocab_size: target vocabulary size, including special and reserved
        tokens; BPE stops early if the corpus has fewer useful merges
    """
    tok = tokenizers.Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size - len(RESERVED_TOKENS),
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    tok.add_tokens(
        [tokenizers.AddedToken(t, normalized=False) for t in RESERVED_TOKENS]
    )
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token=EOS,
        pad_token=PAD,
        clean_up_tokenization_spaces=False,
    )
    return fast


def load_tokenizer(path: str) -> transformers.PreTrainedTokenizerFast:
    """
    Load a tokenizer saved with `save_pretrained`.
    """
    return transformers.PreTrainedTokenizerFast.from_pretrained(path)


def encode_example(tok: transformers.PreTrainedTokenizerFast, text: str) -> List[int]:
    """
    Encode a full training text and append EOS.
    """
    return tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]


def iter_jsonl_texts(paths: List[str], limit: int = 0) -> Iterator[str]:
    """
    Yield prompt + solution texts from JSONL record files.
    """
    n = 0
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                yield r["prompt"] + r["solution"]
                n += 1
                if limit and n >= limit:
                    return


def _main() -> None:
    ap = argparse.ArgumentParser(description="Train the BPE tokenizer.")
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=500_000, help="max texts (0 = all)")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    tok = build_tokenizer(iter_jsonl_texts(a.jsonl, a.limit), a.vocab_size)
    tok.save_pretrained(a.out)
    _LOG.info("Saved tokenizer with %d tokens to %s", len(tok), a.out)


if __name__ == "__main__":
    _main()
