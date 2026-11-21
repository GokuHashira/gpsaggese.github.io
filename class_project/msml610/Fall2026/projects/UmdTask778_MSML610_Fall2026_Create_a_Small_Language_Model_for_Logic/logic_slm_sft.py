"""
Supervised fine-tuning data: prompt/solution pairs with loss masking, and
pass@k evaluation used to pick the SFT checkpoint.

Import as:

import logic_slm_sft as lssft
"""

import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import transformers

import logic_slm_eval as lseval
import logic_slm_format as lsform
import logic_slm_mixture as lsmix
import logic_slm_scorers as lssco

IGNORE_INDEX = -100


# Real SFT data (plan v2): source -> (JSONL glob relative to the project, max examples).
REAL_SFT: Dict[str, Tuple[str, int]] = {
    "openmathinstruct2": ("data/processed/real/openmathinstruct2/*.jsonl.gz", 150_000),
    "metamathqa": ("data/processed/real/metamathqa/*.jsonl.gz", 100_000),
    "tinygsm": ("data/processed/real/tinygsm/*.jsonl.gz", 80_000),
    "orcamath": ("data/processed/real/orcamath/*.jsonl.gz", 60_000),
    "gsm8k_train": ("data/processed/real/gsm8k-train/*.jsonl.gz", 10_000),
}


def load_real_pairs(
    spec: Dict[str, Tuple[str, int]], seed: int, held: Set[str], n_val: int, scale: float = 1.0,
    eval_index: Optional[Set[int]] = None,
) -> Tuple[List[Tuple[str, str]], List[dict], Dict[str, int]]:
    """
    Read real (prompt, solution) pairs from converted JSONL shards.

    Per source, files are visited in a seeded order; the first `n_val` usable
    records become validation records, the next `scale * max` become training
    pairs. Prompts in `held` and records with a 13-gram overlap with the eval
    sets (`eval_index`) are skipped.

    :return: (training pairs, validation records, training count per source)
    """
    import glob
    import gzip
    import json

    import logic_slm_decontam as lsdec

    pairs: List[Tuple[str, str]] = []
    val: List[dict] = []
    counts: Dict[str, int] = {}
    for name in sorted(spec):
        pattern, n_max = spec[name]
        n_train = int(n_max * scale)
        paths = sorted(glob.glob(pattern))
        random.Random(lsmix.stable_seed(seed, "sft-real", name)).shuffle(paths)
        got_val = got = 0
        for path in paths:
            with gzip.open(path, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    if r["prompt"] in held:
                        continue
                    if eval_index and lsdec.is_contaminated(r["prompt"] + r["solution"], eval_index):
                        continue
                    if got_val < n_val:
                        val.append({**r, "gen": name})
                        got_val += 1
                    elif got < n_train:
                        pairs.append((r["prompt"], r["solution"]))
                        got += 1
                    else:
                        break
            if got >= n_train and got_val >= n_val:
                break
        counts[name] = got
    return pairs, val, counts


def build_sft_examples(
    mixture: Dict[str, dict], n_per_unit_share: int, seed: int, held: Set[str],
) -> List[Tuple[str, str]]:
    """
    Generate (prompt, solution) pairs from the generator entries of `mixture`
    (stored real-data entries are skipped; see `load_real_pairs`); generator
    `g` contributes `share_g * n_per_unit_share` examples. Held-out prompts are
    skipped.
    """
    pairs: List[Tuple[str, str]] = []
    for name in sorted(n for n, s in mixture.items() if not lsmix.is_stored(s)):
        n = max(1, int(mixture[name]["share"] * n_per_unit_share))
        got = 0
        for ex in lsmix.generate(mixture, name, 3 * n, lsmix.stable_seed(seed, "sft", name)):
            prompt = lsform.render_prompt(ex)
            if prompt in held:
                continue
            pairs.append((prompt, lsform.render_solution(ex)))
            got += 1
            if got == n:
                break
    random.Random(seed).shuffle(pairs)
    return pairs


class SFTDataset(torch.utils.data.Dataset):
    """
    Tokenized pairs: input = prompt + solution + EOS, labels mask the prompt.
    """

    def __init__(
        self, pairs: Sequence[Tuple[str, str]], tok: transformers.PreTrainedTokenizerFast,
        max_len: int,
    ) -> None:
        prompts = tok([p for p, _ in pairs], add_special_tokens=False)["input_ids"]
        sols = tok([s for _, s in pairs], add_special_tokens=False)["input_ids"]
        self.items: List[Tuple[List[int], int]] = []
        for p, s in zip(prompts, sols):
            ids = p + s + [tok.eos_token_id]
            if len(ids) <= max_len:
                self.items.append((ids, len(p)))
        self.pad_id = tok.pad_token_id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Tuple[List[int], int]:
        return self.items[i]

    def collate(self, batch: List[Tuple[List[int], int]]) -> Dict[str, torch.Tensor]:
        """
        Right-pad a batch; labels are IGNORE_INDEX on prompt and padding.
        """
        n = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), IGNORE_INDEX, dtype=torch.long)
        attn = torch.zeros((len(batch), n), dtype=torch.long)
        for i, (ids, n_prompt) in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            labels[i, n_prompt : len(ids)] = torch.tensor(ids[n_prompt:])
            attn[i, : len(ids)] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def masked_lm_loss(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Mean next-token loss over solution tokens only.
    """
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    shift_logits = logits[:, :-1].float()
    shift_labels = batch["labels"][:, 1:]
    return torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def pass_at_k_eval(
    model: transformers.PreTrainedModel,
    tok: transformers.PreTrainedTokenizerFast,
    records: List[dict],
    *,
    n_samples: int,
    ks: Sequence[int],
    temperature: float = 0.7,
    max_new_tokens: int = 192,
    batch_size: int = 128,
    seed: Optional[int] = 0,
) -> Dict[str, float]:
    """
    Sample `n_samples` completions per record and return pass@k (unbiased)
    plus greedy exact match.

    :param seed: seed for the sampling; the global RNG state is restored
        afterwards, so the evaluation does not reset the training RNG (e.g.
        the dropout masks of the next epoch)
    """
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(seed)
        return _pass_at_k_eval(
            model, tok, records, n_samples=n_samples, ks=ks, temperature=temperature,
            max_new_tokens=max_new_tokens, batch_size=batch_size,
        )


def _pass_at_k_eval(
    model: transformers.PreTrainedModel,
    tok: transformers.PreTrainedTokenizerFast,
    records: List[dict],
    *,
    n_samples: int,
    ks: Sequence[int],
    temperature: float,
    max_new_tokens: int,
    batch_size: int,
) -> Dict[str, float]:
    prompts = [r["prompt"] for r in records]
    golds = [r["answer"] for r in records]
    tasks = [r["task"] for r in records]
    greedy = lseval.generate_completions(
        model, tok, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size
    )
    out = {"greedy_exact_match": sum(
        lssco.exact_match(c, g, t) for c, g, t in zip(greedy, golds, tasks)) / len(records)}
    if n_samples > 0:
        samples = lseval.generate_completions(
            model, tok, [p for p in prompts for _ in range(n_samples)],
            max_new_tokens=max_new_tokens, batch_size=batch_size, do_sample=True,
            temperature=temperature,
        )
        counts = []
        for i, (g, t) in enumerate(zip(golds, tasks)):
            chunk = samples[i * n_samples : (i + 1) * n_samples]
            counts.append(sum(lssco.exact_match(c, g, t) for c in chunk))
        out.update(lssco.pass_at_k_over_problems(counts, n_samples, ks))
    return out
