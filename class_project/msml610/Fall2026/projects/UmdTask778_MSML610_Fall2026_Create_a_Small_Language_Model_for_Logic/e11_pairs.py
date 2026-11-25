#!/usr/bin/env python
"""
Write the training pairs of the matched-compute control (E11): the SAME problem pool GRPO trains on
(same arguments as slurm/grpo.sbatch, same length filter), each with the dataset's own correct
solution. Training on these with supervision, for the GPU time GRPO used, isolates reward vs
supervision (the assigned idea's Q3).

Example:

> python e11_pairs.py --tokenizer models/rft_q125m/best --out results/posttrain/e11_pairs.jsonl
"""

import argparse
import json

import grpo
import logic_slm_posttrain as lspost


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048 - 512)
    a = ap.parse_args(argv)
    import transformers

    ns = argparse.Namespace(problems="", corpus="data/processed/corpus_v2", raw_dir="data/raw",
                            real_eval_dir="data/processed/real/eval", seed=4321, per_real=3000,
                            proofwriter=3000, synth_unit=30000,
                            synth_override='{"m1": {"state_steps": true}}',
                            reuse=["gsm8k_train:7469"], pool_skip=3000, synth_tag="grpo")
    tok = transformers.AutoTokenizer.from_pretrained(a.tokenizer)
    pool, dropped = lspost.filter_by_length(grpo.load_problems(ns), tok, a.max_prompt_tokens)
    rows = [{"prompt": p["prompt"], "solution": p["solution"]} for p in pool if p.get("solution")]
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"E11 pairs: {len(rows)} written ({dropped} too long, {len(pool) - len(rows)} without solution)")
    return 0


if __name__ == "__main__":
    main()
