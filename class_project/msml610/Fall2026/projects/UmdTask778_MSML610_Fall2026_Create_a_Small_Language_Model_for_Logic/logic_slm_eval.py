"""
Generate completions for evaluation prompts and score them.

Import as:

import logic_slm_eval as lseval
"""

import json
import logging
import os
from typing import Dict, List

import torch
import transformers

import logic_slm_data as lsdata
import logic_slm_scorers as lssco

_LOG = logging.getLogger(__name__)


@torch.no_grad()
def generate_completions(
    model: transformers.PreTrainedModel,
    tok: transformers.PreTrainedTokenizerFast,
    prompts: List[str],
    *,
    max_new_tokens: int = 192,
    batch_size: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> List[str]:
    """
    Generate one completion per prompt (greedy by default), stopping at EOS.
    """
    model.eval()
    device = next(model.parameters()).device
    old_side = tok.padding_side
    tok.padding_side = "left"
    outs: List[str] = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=None,
                top_k=None,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1] :]
            outs.extend(tok.batch_decode(new, skip_special_tokens=True))
    finally:
        tok.padding_side = old_side
    return outs


def evaluate_jsonl(
    model: transformers.PreTrainedModel,
    tok: transformers.PreTrainedTokenizerFast,
    path: str,
    *,
    limit: int = 500,
    out_path: str = "",
    **gen_kwargs,
) -> Dict[str, float]:
    """
    Greedy-decode the first `limit` records of `path` and score them.

    Adds accuracy by number of operations / steps when records have `n_ops`.

    :param out_path: optional JSONL file for per-example predictions
    """
    recs = lsdata.read_records(path, limit)
    if not recs:
        _LOG.warning("No records in %s: nothing to evaluate", path)
        return {"n": 0.0}
    task = recs[0]["task"]
    comps = generate_completions(model, tok, [r["prompt"] for r in recs], **gen_kwargs)
    golds = [r["answer"] for r in recs]
    # L1 / L2 steps are symbolic, so steps that do not parse count as invalid.
    symbolic = recs[0].get("level") in ("L1", "L2")
    metrics = lssco.score_completions(comps, golds, task, symbolic=symbolic)
    # Accuracy by difficulty: operations (M1), steps (M2, M3) or proof depth (logic).
    key = next((k for k in ("n_ops", "n_steps", "depth") if k in recs[0]), None)
    if key is not None:
        by: Dict[int, List[bool]] = {}
        for r, c in zip(recs, comps):
            if key in r:
                by.setdefault(r[key], []).append(lssco.exact_match(c, r["answer"], task))
        short = {"n_ops": "ops", "n_steps": "steps", "depth": "depth"}[key]
        for k in sorted(by):
            metrics[f"exact_match@{short}={k}"] = sum(by[k]) / len(by[k])
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            for r, c in zip(recs, comps):
                f.write(json.dumps({"prompt": r["prompt"], "gold": r["answer"], "completion": c}) + "\n")
    return metrics
