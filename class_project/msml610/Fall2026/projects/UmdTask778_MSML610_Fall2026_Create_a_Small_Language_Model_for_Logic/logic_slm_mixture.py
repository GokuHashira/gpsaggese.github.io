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
from typing import Dict, Iterator, List, Optional, Set

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


class MixtureBlocks(torch.utils.data.Dataset):
    """
    Block `j` -> LongTensor of `seq_len + 1` tokens, generated deterministically.
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
        self._tok = None
        # Per-example sampling weights so that *token* shares match `share`.
        self._token_shares = shares / shares.sum()
        self._weights: Optional[List[float]] = None

    def __len__(self) -> int:
        return self.n_blocks

    def _setup(self) -> None:
        if self._tok is not None:
            return
        self._tok = lstok.load_tokenizer(self.tokenizer_dir)
        # Estimate mean tokens per example of each generator from a sample.
        mean_len = []
        for name in self.names:
            lens = [
                len(self._tok.encode(lsform.render_example(e), add_special_tokens=False)) + 1
                for e in generate(self.mixture, name, 200, stable_seed(self.seed, "len", name))
            ]
            mean_len.append(float(np.mean(lens)))
        w = self._token_shares / np.array(mean_len)
        self._weights = list(w / w.sum())

    def _examples(self, block: int) -> Iterator[str]:
        rng = random.Random(stable_seed(self.seed, "block", block))
        iters: Dict[str, Iterator[lsform.Example]] = {}
        while True:
            name = rng.choices(self.names, weights=self._weights)[0]
            if name not in iters:
                iters[name] = iter(
                    generate(self.mixture, name, 10**9, stable_seed(self.seed, "block", block, name))
                )
            ex = next(iters[name])
            prompt = lsform.render_prompt(ex)
            if prompt in self.held:
                continue
            yield prompt + lsform.render_solution(ex)

    def __getitem__(self, block: int) -> torch.Tensor:
        self._setup()
        need = self.seq_len + 1
        ids: List[int] = []
        eos = self._tok.eos_token_id
        texts = self._examples(block)
        while len(ids) < need:
            batch = [next(texts) for _ in range(8)]
            for x in self._tok(batch, add_special_tokens=False)["input_ids"]:
                ids.extend(x)
                ids.append(eos)
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
