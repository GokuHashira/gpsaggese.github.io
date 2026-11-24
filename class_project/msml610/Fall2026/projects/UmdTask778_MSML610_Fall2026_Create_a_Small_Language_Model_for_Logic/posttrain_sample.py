#!/usr/bin/env python
"""
Sample N completions per training problem from a model and score them against
the gold answers (exact match); write the scored samples and the RFT pairs.

The problems come from `logic_slm_posttrain` (problems SFT did not train on:
real sources, ProofWriter, new synthetic ones). Everything is grounded in the
datasets' answers: no model judges anything.

Example:

> python posttrain_sample.py --model models/sftfix_q125m/best --out results/posttrain/q125m \
    --n-samples 8 --per-real 3000 --proofwriter 3000 --synth-unit 30000
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

import logic_slm_mixture as lsmix
import logic_slm_posttrain as lspost
import logic_slm_sft as lssft

_LOG = logging.getLogger("posttrain_sample")


def build_pool(a: argparse.Namespace) -> List[dict]:
    with open(os.path.join(a.corpus, "manifest.json")) as f:
        mixture = json.load(f)["mixture"]
    held = lsmix.load_held_prompts(a.corpus)
    eval_index = None
    if os.path.isdir(a.real_eval_dir):
        import logic_slm_decontam as lsdec

        eval_index = lsdec.load_eval_index(a.real_eval_dir)[0]
    per = {name: a.per_real for name in lssft.REAL_SFT}
    reuse = set()
    for item in (a.reuse or []):
        name, _, n = item.partition(":")
        if name not in lssft.REAL_SFT:
            raise ValueError(f"--reuse: unknown source {name!r}")
        reuse.add(name)
        per[name] = int(n) if n else a.per_real
    pool = lspost.real_pool(lssft.REAL_SFT, a.seed, held, a.sft_val_per_source, a.sft_real_scale,
                            per, eval_index, reuse=reuse,
                            extra_skip=getattr(a, "pool_skip", 0))
    if a.proofwriter > 0:
        import glob

        import logic_slm_data as lsdata
        import logic_slm_proofwriter as lspw

        excl = set()
        for path in glob.glob(os.path.join(a.corpus, "eval", "real_*proofwriter*.jsonl")) + \
                glob.glob(os.path.join(a.corpus, "eval", "real_pw-*.jsonl")):
            excl |= lspw.eval_theories(lsdata.read_records(path))
        pool += lspost.proofwriter_pool(a.raw_dir, a.proofwriter,
                                        a.sft_proofwriter + 100 + getattr(a, "pool_skip", 0),
                                        a.seed, excl)
    if a.synth_unit > 0:
        overrides = json.loads(a.synth_override) if a.synth_override else None
        pool += lspost.synthetic_pool(mixture, a.synth_unit, a.seed, held, overrides,
                                      tag=getattr(a, "synth_tag", "posttrain"))
    return pool


def main(argv: List[str] = None) -> Dict[str, float]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--corpus", default="data/processed/corpus_v2")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--real-eval-dir", default="data/processed/real/eval")
    ap.add_argument("--seed", type=int, default=4321, help="same seed as SFT (to skip its data)")
    ap.add_argument("--sft-val-per-source", type=int, default=100)
    ap.add_argument("--sft-real-scale", type=float, default=1.0)
    ap.add_argument("--sft-proofwriter", type=int, default=60000)
    ap.add_argument("--per-real", type=int, default=3000, help="problems per real source")
    ap.add_argument("--reuse", nargs="*", default=[],
                    help="real sources sampled even though SFT trained on them, as name[:count], "
                         "e.g. gsm8k_train:7469 (tagged seen_in_sft; reported separately)")
    ap.add_argument("--proofwriter", type=int, default=3000)
    ap.add_argument("--synth-unit", type=int, default=30000,
                    help="synthetic problems per unit of share (shares sum to ~0.1)")
    ap.add_argument("--synth-override", default='{"m1": {"state_steps": true}}')
    ap.add_argument("--pool-skip", type=int, default=0,
                    help="skip this many more records per source (problems an earlier stage used)")
    ap.add_argument("--synth-tag", default="posttrain", help="seed tag for new synthetic problems")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--backend", choices=("auto", "vllm", "hf"), default="auto")
    ap.add_argument("--batch-size", type=int, default=128, help="HF backend only")
    ap.add_argument("--rft-max-per-problem", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N problems")
    a = ap.parse_args(argv)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    pool = build_pool(a)
    if a.limit:
        pool = pool[: a.limit]
    by: Dict[str, int] = {}
    for p in pool:
        by[p["gen"]] = by.get(p["gen"], 0) + 1
    _LOG.info("problem pool: %d %s (%.0fs)", len(pool), by, time.time() - t0)
    backend = a.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401

            backend = "vllm"
        except ImportError:
            backend = "hf"
    t1 = time.time()
    prompts = [p["prompt"] for p in pool]
    if backend == "vllm":
        comps = lspost.sample_vllm(a.model, prompts, a.n_samples, a.temperature, a.max_new_tokens,
                                   seed=a.seed)
    else:
        comps = lspost.sample_hf(a.model, prompts, a.n_samples, a.temperature, a.max_new_tokens,
                                 batch_size=a.batch_size, seed=a.seed)
    sample_s = time.time() - t1
    samples = lspost.score_samples(pool, comps)
    lspost.write_jsonl(os.path.join(a.out, "samples.jsonl"), samples)
    pairs = lspost.rft_pairs(samples, a.rft_max_per_problem, seed=a.seed)
    lspost.write_jsonl(os.path.join(a.out, "rft_pairs.jsonl"),
                       [{"prompt": p, "solution": c} for p, c in pairs])
    summary = lspost.summarize(samples)
    n_tok = sum(len(c) for cs in comps for c in cs) / 4  # rough characters -> tokens
    summary.update({"backend": backend, "n_problems": len(pool), "n_samples": a.n_samples,
                    "sample_time_s": sample_s, "rft_pairs": len(pairs),
                    "approx_tokens_per_s": n_tok / max(sample_s, 1e-9), "model": a.model,
                    "pool_by_source": by})
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _LOG.info("summary: %s", {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in summary.items() if not isinstance(v, dict)})
    return summary


if __name__ == "__main__":
    main()
