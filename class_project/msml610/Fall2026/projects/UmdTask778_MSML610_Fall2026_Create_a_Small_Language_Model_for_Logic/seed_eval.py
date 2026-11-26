#!/usr/bin/env python
"""
Sampling metrics over several seeds, with vLLM (fast generation).

For one model: for each seed, sample `n` answers per problem on each eval set and compute pass@1,
pass@n and majority@n; then report the mean and the spread over seeds. This shows how much of a
difference in the sampled metrics is seed noise.

> python seed_eval.py --model <vllm-ready model dir> --name grpo2_q125m --out results/seeds/grpo2_q125m \
    --seeds 0 1 2
"""

import argparse
import collections
import json
import os
import statistics
from typing import Dict, List, Sequence

import logic_slm_data as lsdata
import logic_slm_format as lsform
import logic_slm_scorers as lssco

DEFAULT_SETS = ["real_gsm8k-test", "real_svamp", "real_pw-depth5"]


def majority_answer(completions: Sequence[str], task: str) -> str:
    """
    Most common normalized final answer among the samples ("" if none has one).
    """
    norm = [lsform.normalize_answer(lsform.parse_answer(c), task) for c in completions]
    norm = [a for a in norm if a]
    return collections.Counter(norm).most_common(1)[0][0] if norm else ""


def sample_metrics(recs: Sequence[dict], samples: Sequence[Sequence[str]]) -> Dict[str, float]:
    """
    pass@1, pass@n and majority@n for `n` sampled completions per problem.
    """
    task = recs[0]["task"]
    n = len(samples[0])
    counts = [sum(lssco.exact_match(c, r["answer"], task) for c in comps) for r, comps in zip(recs, samples)]
    out = lssco.pass_at_k_over_problems(counts, n, sorted({1, n}))
    votes = [majority_answer(comps, task) for comps in samples]
    out[f"majority@{n}"] = sum(bool(v) and lsform.answers_match(v, r["answer"], task)
                               for v, r in zip(votes, recs)) / len(recs)
    return out


def aggregate(per_seed: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Mean, standard deviation (0 for one seed), min and max of every metric over the seeds.
    """
    keys = sorted(next(iter(per_seed.values())))
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [m[k] for m in per_seed.values()]
        out[k] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                  "min": min(vals), "max": max(vals), "n_seeds": len(vals)}
    return out


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir loadable by vLLM")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-dir", default="data/processed/corpus_v2/eval")
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-memory", type=float, default=0.85)
    a = ap.parse_args(argv)
    import vllm

    llm = vllm.LLM(model=a.model, dtype="bfloat16", seed=0, enforce_eager=True,
                   gpu_memory_utilization=a.gpu_memory, max_model_len=2048)
    os.makedirs(a.out, exist_ok=True)
    res: Dict[str, dict] = {}
    for set_name in a.sets:
        recs = lsdata.read_records(os.path.join(a.eval_dir, f"{set_name}.jsonl"), a.limit)
        prompts = [r["prompt"] for r in recs]
        per_seed: Dict[int, Dict[str, float]] = {}
        for seed in a.seeds:
            params = vllm.SamplingParams(temperature=a.temperature, max_tokens=a.max_new_tokens,
                                         n=a.n_samples, seed=seed)
            outs = llm.generate(prompts, params)
            per_seed[seed] = sample_metrics(recs, [[c.text for c in o.outputs] for o in outs])
            print(a.name, set_name, "seed", seed, {k: round(v, 4) for k, v in per_seed[seed].items()}, flush=True)
        res[set_name] = {"n": len(recs), "per_seed": {str(s): m for s, m in per_seed.items()},
                         "over_seeds": aggregate(per_seed)}
        with open(os.path.join(a.out, "seed_metrics.json"), "w") as f:
            json.dump({"name": a.name, "model": a.model, "n_samples": a.n_samples, "temperature": a.temperature,
                       "seeds": a.seeds, "sets": res}, f, indent=2)


if __name__ == "__main__":
    main()
