"""
Generator mixture and an on-the-fly pretraining dataset.

`MixtureBlocks` is a map-style dataset: block `j` is generated from a seed
derived from `(seed, j)`, so

- every model size sees exactly the same blocks in the same order,
- training can resume at any step without replaying data,
- no corpus has to be stored on disk.

Each block is filled with whole examples (prompt + solution + EOS) until it
holds `seq_len + 1` tokens; the tail of the last example is dropped. Examples
whose prompt is in the held-out set (eval / validation) are skipped.

Import as:

import logic_slm_mixture as lsmix
"""

import hashlib
import importlib
import json
import random
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch

import logic_slm_format as lsform
import logic_slm_tokenizer as lstok

# Generator registry: module, token share, kwargs per split. `train` kwargs
# are used for training blocks, validation and the tokenizer sample.
#
# Training covers 1-4 reasoning steps; `test_extrap` uses 5-8 steps and
# `test_heldout` (M3, L3) uses templates never seen in training.
MIXTURE: Dict[str, dict] = {
    "m1": {
        "module": "logic_slm_math_m1",
        "share": 0.15,
        "train": {"n_ops_range": (1, 4)},
        "eval": {"test_id": {"n_ops_range": (1, 4)}, "test_extrap": {"n_ops_range": (5, 8)}},
    },
    "m2": {
        "module": "logic_slm_math_m2",
        "share": 0.15,
        "train": {"n_steps_range": (1, 4)},
        "eval": {"test_id": {"n_steps_range": (1, 4)}, "test_extrap": {"n_steps_range": (5, 8)}},
    },
    "m3": {
        "module": "logic_slm_math_m3",
        "share": 0.20,
        "train": {"n_steps_range": (1, 4), "n_distractors_range": (0, 2)},
        "eval": {
            "test_id": {"n_steps_range": (1, 4), "n_distractors_range": (0, 2)},
            "test_extrap": {"n_steps_range": (5, 8), "n_distractors_range": (0, 2)},
            "test_heldout": {"n_steps_range": (1, 4), "n_distractors_range": (0, 2),
                             "template_split": "heldout"},
        },
    },
    "l1": {
        "module": "logic_slm_logic",
        "share": 0.15,
        "train": {"level": "L1", "depth_range": (1, 4)},
        "eval": {"test_id": {"level": "L1", "depth_range": (1, 4)},
                 "test_extrap": {"level": "L1", "depth_range": (5, 8)}},
    },
    "l2": {
        "module": "logic_slm_logic",
        "share": 0.15,
        "train": {"level": "L2", "depth_range": (1, 4)},
        "eval": {"test_id": {"level": "L2", "depth_range": (1, 4)},
                 "test_extrap": {"level": "L2", "depth_range": (5, 8)}},
    },
    "l3": {
        "module": "logic_slm_logic",
        "share": 0.20,
        "train": {"level": "L3", "depth_range": (1, 4)},
        "eval": {"test_id": {"level": "L3", "depth_range": (1, 4)},
                 "test_heldout": {"level": "L3", "depth_range": (1, 4),
                                  "template_split": "heldout"}},
    },
}


# Plan v2 (references/project_plan_v2.md): real data 90%, synthetic 10%. Stored
# entries point to pre-tokenized sources inside the corpus dir (built by
# build_corpus_v2.py); the first `holdout` documents of each are validation only.
_HOLDOUT = 2000
MIXTURE_V2: Dict[str, dict] = {
    # Shares set from the measured token counts (corpus v2, 2026-11-19) so that no
    # source exceeds ~4 passes even for the largest model (12B tokens).
    "finemath": {"stored": "sources/finemath", "share": 0.46, "holdout": _HOLDOUT},
    "tinygsm": {"stored": "sources/tinygsm", "share": 0.20, "holdout": _HOLDOUT},
    "openmathinstruct2": {"stored": "sources/openmathinstruct2", "share": 0.19, "holdout": _HOLDOUT},
    "metamathqa": {"stored": "sources/metamathqa", "share": 0.03, "holdout": _HOLDOUT},
    "orcamath": {"stored": "sources/orcamath", "share": 0.0196, "holdout": _HOLDOUT},
    "gsm8k_train": {"stored": "sources/gsm8k_train", "share": 0.0004, "holdout": 500},
    **{name: {**spec, "share": share} for (name, spec), share in zip(
        MIXTURE.items(), (0.015, 0.015, 0.02, 0.015, 0.015, 0.02))},
}


def stable_seed(*parts) -> int:
    """
    Deterministic 63-bit seed from arbitrary parts (independent of PYTHONHASHSEED).
    """
    h = hashlib.blake2b("/".join(map(str, parts)).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little") >> 1


def generate(mixture: Dict[str, dict], name: str, n: int, seed: int, split: str = "train"):
    """
    Iterate `n` examples of generator `name` for a split ("train" or an eval
    split name).
    """
    spec = mixture[name]
    kwargs = spec["train"] if split == "train" else spec["eval"][split]
    mod = importlib.import_module(spec["module"])
    return mod.generate_dataset(n, seed, **kwargs)


def to_record(mixture: Dict[str, dict], name: str, ex: lsform.Example) -> dict:
    mod = importlib.import_module(mixture[name]["module"])
    return {"gen": name, **mod.example_to_record(ex)}


def load_mixture(path: str) -> Dict[str, dict]:
    """
    Load a mixture from JSON (as saved in a corpus manifest), or return the
    default `MIXTURE` when `path` is empty.
    """
    if not path:
        return MIXTURE
    with open(path) as f:
        data = json.load(f)
    return data.get("mixture", data)


class StoredSource:
    """
    A pre-tokenized real-data source: `tokens.bin` (uint16 token ids, each
    document ends with EOS) and `offsets.npy` (int64 document start offsets,
    length n_docs + 1). The first `holdout` documents are reserved for
    validation and never sampled for training.
    """

    def __init__(self, path: str, holdout: int = 0) -> None:
        import os

        self.path = path
        self.tokens = np.memmap(os.path.join(path, "tokens.bin"), dtype=np.uint16, mode="r")
        self.offsets = np.load(os.path.join(path, "offsets.npy"), mmap_mode="r")
        self.n_docs = len(self.offsets) - 1
        self.holdout = min(holdout, max(self.n_docs - 1, 0))
        if self.n_docs - self.holdout < 1:
            raise ValueError(f"No training documents in {path}")

    def doc(self, i: int) -> np.ndarray:
        return self.tokens[int(self.offsets[i]) : int(self.offsets[i + 1])]

    def mean_len(self, cap: int, rng: random.Random, n: int = 2000) -> float:
        """
        Mean training-document length (each capped at `cap`) from a sample.
        """
        idx = [rng.randrange(self.holdout, self.n_docs) for _ in range(n)]
        return float(np.mean([min(int(self.offsets[i + 1] - self.offsets[i]), cap) for i in idx]))


def is_stored(spec: dict) -> bool:
    return "stored" in spec


class MixtureBlocks(torch.utils.data.Dataset):
    """
    Block `j` -> LongTensor of `seq_len + 1` tokens, built deterministically
    from `(seed, j)` out of generator sources (generated on the fly) and stored
    sources (pre-tokenized real data, entries with a `stored` path relative to
    `corpus_dir`). Sources are drawn per document with weights chosen so that
    *token* shares match each entry's `share`. Documents longer than the space
    left contribute a random window; the tail of the last document is dropped.
    """

    def __init__(
        self,
        mixture: Dict[str, dict],
        tokenizer_dir: str,
        seq_len: int,
        seed: int,
        *,
        held_prompts: Optional[Set[str]] = None,
        n_blocks: int = 10**12,
        corpus_dir: str = "",
    ) -> None:
        self.mixture = mixture
        self.names = sorted(mixture)
        shares = np.array([mixture[n]["share"] for n in self.names], dtype=float)
        if (shares <= 0).any():
            raise ValueError("All shares must be positive")
        self.tokenizer_dir = tokenizer_dir
        self.seq_len = seq_len
        self.seed = seed
        self.held = held_prompts or set()
        self.n_blocks = n_blocks
        self.corpus_dir = corpus_dir
        self._tok = None
        self._stored: Dict[str, StoredSource] = {}
        # Per-document sampling weights so that *token* shares match `share`.
        self._token_shares = shares / shares.sum()
        self._weights: Optional[List[float]] = None

    def __len__(self) -> int:
        return self.n_blocks

    def _setup(self) -> None:
        if self._tok is not None:
            return
        import os

        self._tok = lstok.load_tokenizer(self.tokenizer_dir)
        need = self.seq_len + 1
        mean_len = []
        for name in self.names:
            spec = self.mixture[name]
            rng = random.Random(stable_seed(self.seed, "len", name))
            if is_stored(spec):
                src = StoredSource(os.path.join(self.corpus_dir, spec["stored"]), spec.get("holdout", 0))
                self._stored[name] = src
                mean_len.append(src.mean_len(need, rng))
            else:
                lens = [
                    len(self._tok.encode(lsform.render_example(e), add_special_tokens=False)) + 1
                    for e in generate(self.mixture, name, 200, stable_seed(self.seed, "len", name))
                ]
                mean_len.append(float(np.mean(lens)))
        w = self._token_shares / np.array(mean_len)
        self._weights = list(w / w.sum())

    def _docs(self, block: int, rng: random.Random) -> Iterator[Tuple[str, object]]:
        """
        Yield ("tokens", np.ndarray) for stored sources and ("text", str) for generators.
        """
        iters: Dict[str, Iterator[lsform.Example]] = {}
        while True:
            name = rng.choices(self.names, weights=self._weights)[0]
            if name in self._stored:
                src = self._stored[name]
                yield "tokens", src.doc(rng.randrange(src.holdout, src.n_docs))
                continue
            if name not in iters:
                iters[name] = iter(
                    generate(self.mixture, name, 10**9, stable_seed(self.seed, "block", block, name))
                )
            ex = next(iters[name])
            prompt = lsform.render_prompt(ex)
            if prompt in self.held:
                continue
            yield "text", prompt + lsform.render_solution(ex)

    def __getitem__(self, block: int) -> torch.Tensor:
        self._setup()
        need = self.seq_len + 1
        ids: List[int] = []
        eos = self._tok.eos_token_id
        rng = random.Random(stable_seed(self.seed, "block", block))
        docs = self._docs(block, rng)
        while len(ids) < need:
            kind, doc = next(docs)
            if kind == "text":
                toks = self._tok.encode(doc, add_special_tokens=False) + [eos]
            else:
                toks = doc
                room = need - len(ids)
                if len(toks) > room and len(toks) > need:
                    # Long document: take a random window instead of always its start.
                    start = rng.randrange(0, len(toks) - room + 1)
                    toks = toks[start : start + room]
                toks = [int(t) for t in toks]
            ids.extend(toks)
        return torch.tensor(ids[:need], dtype=torch.long)


def load_held_prompts(corpus_dir: str) -> Set[str]:
    """
    Prompts of all eval and validation records in a corpus directory.
    """
    import glob
    import os

    held: Set[str] = set()
    paths = glob.glob(os.path.join(corpus_dir, "eval", "*.jsonl"))
    paths.append(os.path.join(corpus_dir, "val.jsonl"))
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                held.update(json.loads(line)["prompt"] for line in f)
    return held
