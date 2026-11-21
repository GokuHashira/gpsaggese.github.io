#!/usr/bin/env python
"""
Evaluate a saved model on every test set and record quality and cost.

Used after every stage (pretraining, SFT, post-training) so stages and sizes
are compared with identical measurements. For each eval set it records:

- greedy: exact match, answer-format rate, step validity (synthetic sets),
  accuracy by difficulty, mean solution length (tokens);
- sampling (optional): pass@k (unbiased) and majority-vote accuracy over the
  same k samples;
- inference cost: generated tokens, generation tokens/s, seconds per problem,
  and SU per correct answer.

The run (GPU trace, timings, GPU-hours, SU, energy) is appended to the run
ledger `results/runs.jsonl`; full results go to `<out>/eval_metrics.json` and
per-example predictions to `<out>/pred_<set>.jsonl`.

Example:

> python evaluate.py --model models/pt_125m/final --name pt_125m --stage pretrain \
    --eval-dir data/processed/corpus_v1/eval --out results/eval/pt_125m
"""

import argparse
import collections
import fnmatch
import glob
import json
import logging
import os
import sys
import time
from typing import Dict, List

import torch
import transformers

import logic_slm_data as lsdata
import logic_slm_eval as lseval
import logic_slm_format as lsform
import logic_slm_metrics as lsmet
import logic_slm_scorers as lssco
import logic_slm_tokenizer as lstok
import logic_slm_train as lstrain

_LOG = logging.getLogger("evaluate")


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="model dir (save_pretrained)")
    ap.add_argument("--name", required=True, help="run name, e.g. pt_125m or sft_358m")
    ap.add_argument("--stage", required=True, help="pretrain / sft / rft / grpo / ...")
    ap.add_argument("--eval-dir", nargs="+", required=True, help="dirs with *.jsonl eval sets")
    ap.add_argument("--sets", nargs="*", default=["*"],
                    help="only eval sets whose name matches one of these glob patterns (e.g. 'real_*')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500, help="problems per eval set (greedy)")
    ap.add_argument("--n-samples", type=int, default=8, help="samples per problem (0 = greedy only)")
    ap.add_argument("--sample-limit", type=int, default=200, help="problems per set for sampling")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ledger", default=lsmet.DEFAULT_LEDGER)
    ap.add_argument("--su-per-gpu-hour", type=float, default=0.0)
    return ap.parse_args(argv)


def su_per_correct(cost: Dict[str, float], n_correct: float):
    """
    SU spent in the evaluation phase per correctly answered problem (None if none correct).
    """
    return cost.get("su_eval", 0.0) / n_correct if n_correct else None


def _majority(answers: List[str], task: str) -> str:
    norm = [lsform.normalize_answer(a, task) for a in answers if a is not None]
    norm = [a for a in norm if a]
    return collections.Counter(norm).most_common(1)[0][0] if norm else ""


def eval_set(model, tok, path: str, a: argparse.Namespace, out_dir: str) -> Dict[str, float]:
    """
    Greedy and sampled evaluation of one JSONL eval set.
    """
    name = os.path.basename(path)[:-6]
    recs = lsdata.read_records(path, a.limit)
    task = recs[0]["task"]
    prompts = [r["prompt"] for r in recs]
    golds = [r["answer"] for r in recs]
    # Greedy.
    t0 = time.time()
    comps = lseval.generate_completions(model, tok, prompts, max_new_tokens=a.max_new_tokens,
                                        batch_size=a.batch_size)
    t_greedy = time.time() - t0
    m = lssco.score_completions(comps, golds, task)
    gen_tokens = sum(len(tok.encode(c, add_special_tokens=False)) for c in comps)
    m.update({
        "greedy_time_s": t_greedy,
        "greedy_gen_tokens": gen_tokens,
        "greedy_tokens_per_s": gen_tokens / max(t_greedy, 1e-9),
        "greedy_s_per_problem": t_greedy / len(recs),
        "mean_solution_tokens": gen_tokens / len(recs),
    })
    key = next((k for k in ("n_ops", "n_steps", "depth") if k in recs[0]), None)
    if key is not None:
        by: Dict[int, List[bool]] = {}
        for r, c in zip(recs, comps):
            by.setdefault(r[key], []).append(lssco.exact_match(c, r["answer"], task))
        short = {"n_ops": "ops", "n_steps": "steps", "depth": "depth"}[key]
        for k in sorted(by):
            m[f"exact_match@{short}={k}"] = sum(by[k]) / len(by[k])
    with open(os.path.join(out_dir, f"pred_{name}.jsonl"), "w") as f:
        for r, c in zip(recs, comps):
            f.write(json.dumps({"prompt": r["prompt"], "gold": r["answer"], "completion": c,
                                "correct": lssco.exact_match(c, r["answer"], task)}) + "\n")
    # Sampling: pass@k and majority vote over the same samples.
    if a.n_samples > 0:
        sub = recs[: a.sample_limit]
        t0 = time.time()
        samples = lseval.generate_completions(
            model, tok, [r["prompt"] for r in sub for _ in range(a.n_samples)],
            max_new_tokens=a.max_new_tokens, batch_size=a.batch_size, do_sample=True,
            temperature=a.temperature)
        t_sample = time.time() - t0
        counts, maj = [], []
        for i, r in enumerate(sub):
            chunk = samples[i * a.n_samples : (i + 1) * a.n_samples]
            counts.append(sum(lssco.exact_match(c, r["answer"], task) for c in chunk))
            vote = _majority([lsform.parse_answer(c) for c in chunk], task)
            maj.append(lsform.answers_match(vote, r["answer"], task) if vote else False)
        ks = sorted({1, 2, 4, 8, 16, a.n_samples} & set(range(1, a.n_samples + 1)))
        m.update(lssco.pass_at_k_over_problems(counts, a.n_samples, ks))
        m[f"majority@{a.n_samples}"] = sum(maj) / len(maj)
        stoks = sum(len(tok.encode(c, add_special_tokens=False)) for c in samples)
        m.update({"sample_n_problems": len(sub), "sample_time_s": t_sample,
                  "sample_gen_tokens": stoks, "sample_tokens_per_s": stoks / max(t_sample, 1e-9)})
    _LOG.info("%s: %s", name, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    return m


def main(argv: List[str] = None) -> Dict[str, Dict[str, float]]:
    a = parse_args(argv)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    timer = lsmet.Timer()
    mon = lsmet.GpuMonitor(a.out).start()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with timer.phase("load"):
        tok = lstok.load_tokenizer(a.model)
        model = transformers.AutoModelForCausalLM.from_pretrained(a.model).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    paths = sorted(
        p for d in a.eval_dir for p in glob.glob(os.path.join(d, "*.jsonl"))
        if any(fnmatch.fnmatch(os.path.basename(p)[:-6], pat) for pat in a.sets)
    )
    if not paths:
        raise FileNotFoundError(f"No eval sets match {a.sets} in {a.eval_dir}")
    results: Dict[str, Dict[str, float]] = {}
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16)
    with timer.phase("eval"), amp:
        for p in paths:
            results[os.path.basename(p)[:-6]] = eval_set(model, tok, p, a, a.out)
            # Save after every set so a later failure never loses finished work.
            with open(os.path.join(a.out, "eval_metrics.partial.json"), "w") as f:
                json.dump(results, f, indent=2)
    gpu = lstrain.gpu_name()
    n_gpus = 1 if device.type == "cuda" else 0
    rate = lstrain.su_rate(a.su_per_gpu_hour)
    cost = lsmet.cost_fields(timer.total(), timer.phases, max(n_gpus, 1), rate)
    n_correct = sum(r["exact_match"] * r["n"] for r in results.values())
    summary = {
        "kind": "eval", "name": a.name, "stage": a.stage, "model": a.model, "params_total": params,
        "gpu": gpu, "git_commit": lsmet.git_commit(os.path.dirname(os.path.abspath(__file__))),
        **lsmet.slurm_info(), **cost, **mon.stop(),
        "su_per_correct_greedy": su_per_correct(cost, n_correct),
        "eval_sets": {k: {kk: v for kk, v in r.items() if not kk.startswith("exact_match@")}
                      for k, r in results.items()},
    }
    with open(os.path.join(a.out, "eval_metrics.json"), "w") as f:
        json.dump({"summary": summary, "per_set": results}, f, indent=2, default=str)
    lsmet.RunLedger(a.ledger).append(summary)
    _LOG.info("done: %s GPU-h, %s SU, %s kWh", round(cost["gpu_hours"], 3), round(cost["su"], 1),
              round(summary["energy_kwh"], 3))
    return results


if __name__ == "__main__":
    main()
