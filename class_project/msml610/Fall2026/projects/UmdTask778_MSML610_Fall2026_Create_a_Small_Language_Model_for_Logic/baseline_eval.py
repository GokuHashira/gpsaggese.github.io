#!/usr/bin/env python
"""
Evaluate open pretrained models (baselines) on our real benchmarks with our own scorer.

The baselines were not trained on our format, so each problem is preceded by a few worked examples in
that format (few-shot): 8 GSM8K train problems for the math sets, 6 ProofWriter train problems (2 per
answer) for ProofWriter. The model continues the text; its completion is cut after the final answer
line (`#### ...`) and scored by exact match, exactly like our own models. Greedy decoding only.

Baselines are reference points only: they never judge, filter or generate training data.

> python baseline_eval.py --model models/baselines/SmolLM2-360M --name base_smollm2-360m --out results/eval/base_smollm2-360m
"""

import argparse
import glob
import gzip
import json
import logging
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import torch
import transformers

import logic_slm_data as lsdata
import logic_slm_format as lsform
import logic_slm_scorers as lssco

_LOG = logging.getLogger(__name__)
# (eval file, output sub-folder, shot kind, max new tokens): the sub-folders match evaluate.py's runs.
SETS = [("real_gsm8k-test", "real", "math", 320), ("real_svamp", "real", "math", 320),
        ("real_pw-depth5", "pw-depth5", "logic", 384)]


def fewshot_prefix(shots: Sequence[Tuple[str, str]]) -> str:
    """
    Worked examples (prompt + solution), separated by a blank line and ending with one.
    """
    return "".join(f"{p}{s.strip()}\n\n" for p, s in shots)


def trim_completion(text: str) -> str:
    """
    Keep the completion up to and including its first final-answer line; a model that goes on to
    write the next problem is cut before it.
    """
    out: List[str] = []
    for line in text.split("\n"):
        if line.startswith(("Problem:", "Premises:")):
            break
        out.append(line)
        if line.startswith("####"):
            break
    return "\n".join(out).strip()


def math_shots(gsm8k_train_dir: str, n: int) -> List[Tuple[str, str]]:
    path = sorted(glob.glob(os.path.join(gsm8k_train_dir, "part-*.jsonl.gz")))[0]
    shots: List[Tuple[str, str]] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            shots.append((r["prompt"], r["solution"]))
            if len(shots) == n:
                break
    return shots


def balanced_shots(pairs: Sequence[Tuple[str, str]], per_answer: int) -> List[Tuple[str, str]]:
    """
    `per_answer` examples of typical length for each final answer, interleaved by answer.

    Examples are taken from the middle of the length ranking (median upwards): the shortest ones
    have trivial proofs and would be poor demonstrations for multi-step problems.
    """
    by: Dict[str, List[Tuple[str, str]]] = {}
    for p, s in pairs:
        by.setdefault(lsform.parse_answer(s) or "", []).append((p, s))
    cols = []
    for k, v in sorted(by.items()):
        if not k:
            continue
        v = sorted(v, key=lambda x: len(x[0]) + len(x[1]))
        start = min(len(v) // 2, max(0, len(v) - per_answer))
        cols.append(v[start : start + per_answer])
    return [c[i] for i in range(per_answer) for c in cols if i < len(c)]


def logic_shots(raw_dir: str, eval_path: str, per_answer: int, seed: int) -> List[Tuple[str, str]]:
    import logic_slm_proofwriter as lspw

    excl = lspw.eval_theories(lsdata.read_records(eval_path))
    pairs, _, _ = lspw.load_sft_pairs(raw_dir, 300, 0, seed, excl)
    return balanced_shots(pairs, per_answer)


class _AnswerLineDone(transformers.StoppingCriteria):
    """Stop when every row has written a complete `#### ...` line (or started a new problem)."""

    def __init__(self, tok, prompt_len: int):
        self.tok, self.prompt_len = tok, prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for text in self.tok.batch_decode(input_ids[:, self.prompt_len:], skip_special_tokens=True):
            i = text.find("####")
            if not ((i >= 0 and "\n" in text[i:]) or "\nProblem:" in text or "\nPremises:" in text):
                return False
        return True


@torch.no_grad()
def generate(model, tok, prompts: List[str], max_new_tokens: int, batch_size: int) -> List[str]:
    device = next(model.parameters()).device
    tok.padding_side = "left"
    outs: List[str] = []
    # Longest prompts first, so an out-of-memory error shows up at once; restored to input order.
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        n = enc["input_ids"].shape[1]
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id,
                             stopping_criteria=transformers.StoppingCriteriaList([_AnswerLineDone(tok, n)]))
        outs.extend(tok.batch_decode(gen[:, n:], skip_special_tokens=True))
    res = [""] * len(prompts)
    for i, o in zip(order, outs):
        res[i] = o
    return res


def main(argv: List[str] = None) -> Dict[str, Dict[str, float]]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local folder of a Hugging Face causal LM")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-dir", default="data/processed/corpus_v2/eval")
    ap.add_argument("--gsm8k-train", default="data/processed/real/gsm8k-train")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--math-shots", type=int, default=8)
    ap.add_argument("--logic-shots-per-answer", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="problems per set (0 = all)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    max_ctx = getattr(model.config, "max_position_embeddings", 0) or 0
    prefixes: Dict[str, str] = {}
    by_part: Dict[str, Dict[str, Dict[str, float]]] = {}
    for set_name, part, kind, max_new in SETS:
        path = os.path.join(a.eval_dir, f"{set_name}.jsonl")
        recs = lsdata.read_records(path, a.limit)
        if kind not in prefixes:
            shots = (math_shots(a.gsm8k_train, a.math_shots) if kind == "math"
                     else logic_shots(a.raw_dir, path, a.logic_shots_per_answer, a.seed))
            prefixes[kind] = fewshot_prefix(shots)
            _LOG.info("%s: %d shots, %d prefix tokens", kind, len(shots), len(tok(prefixes[kind])["input_ids"]))
        prompts = [prefixes[kind] + r["prompt"] for r in recs]
        longest = max(len(x) for x in tok(prompts)["input_ids"])
        if max_ctx and longest + max_new > max_ctx:
            raise ValueError(f"{set_name}: {longest} prompt tokens + {max_new} exceed the context {max_ctx}")
        t0 = time.time()
        comps = [trim_completion(c) for c in generate(model, tok, prompts, max_new, a.batch_size)]
        task = recs[0]["task"]
        correct = [lssco.exact_match(c, r["answer"], task) for c, r in zip(comps, recs)]
        m = {"n": float(len(recs)), "exact_match": sum(correct) / len(recs),
             "has_answer": sum(lsform.parse_answer(c) is not None for c in comps) / len(recs),
             "greedy_time_s": time.time() - t0, "max_prompt_tokens": float(longest)}
        out_dir = os.path.join(a.out, part)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"pred_{set_name}.jsonl"), "w") as f:
            for r, c, ok in zip(recs, comps, correct):
                f.write(json.dumps({"prompt": r["prompt"], "gold": r["answer"], "completion": c, "correct": ok}) + "\n")
        by_part.setdefault(part, {})[set_name] = m
        _LOG.info("%s: %s", set_name, {k: round(v, 4) for k, v in m.items()})
        summary = {"kind": "baseline_eval", "name": a.name, "stage": "baseline", "model": a.model,
                   "params_total": params, "few_shot": {"math": a.math_shots, "logic": 3 * a.logic_shots_per_answer},
                   "decoding": "greedy"}
        with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
            json.dump({"summary": summary, "per_set": by_part[part]}, f, indent=2)
    return {k: v for p in by_part.values() for k, v in p.items()}


if __name__ == "__main__":
    main()
