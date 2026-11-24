#!/usr/bin/env python
"""
Compare evaluation runs (`results/eval/<name>/...`) and break down failures.

For every run name given, it reads all `eval_metrics.json` files under
`results/eval/<name>/` and prints, per eval set: greedy accuracy, pass@1/8,
majority@8, the share of answers with a final `####` line and with valid
steps. Then, from the saved predictions, it classifies the wrong greedy
answers:

- no_answer: no `####` line (truncated or rambling)
- loop: the same line repeated >= 4 times
- wrong_answer: a final answer that is not the gold one

and reports ProofWriter accuracy per proof depth and the predicted-label
distribution, synthetic accuracy per number of steps, and (with `--sft`) the
per-epoch validation history of SFT runs.

Example:

> python analyze_eval.py pt2_q125m sft2_q125m --sft models/sft2_q125m
"""

import argparse
import collections
import glob
import json
import os
from typing import Dict, List, Tuple

EVAL_ROOT = "results/eval"
EVAL_DIR = "data/processed/corpus_v2/eval"
COLS = ("exact_match", "pass@1", "pass@8", "majority@8", "has_answer", "steps_valid")


def load_metrics(name: str) -> Dict[str, Dict[str, float]]:
    """
    {set: metrics} over all sub-directories of a run (later dirs override).
    """
    out: Dict[str, Dict[str, float]] = {}
    for path in sorted(glob.glob(os.path.join(EVAL_ROOT, name, "*", "eval_metrics.json"))):
        sub = os.path.basename(os.path.dirname(path))
        for s, m in json.load(open(path))["per_set"].items():
            key = f"{s}@1024" if sub == "math1024" else s
            out[key] = m
    return out


def load_preds(name: str) -> Dict[str, Tuple[str, List[dict]]]:
    out = {}
    for path in sorted(glob.glob(os.path.join(EVAL_ROOT, name, "*", "pred_*.jsonl"))):
        sub = os.path.basename(os.path.dirname(path))
        s = os.path.basename(path)[5:-6]
        key = f"{s}@1024" if sub == "math1024" else s
        out[key] = (sub, [json.loads(line) for line in open(path)])
    return out


def is_loop(text: str, n: int = 4) -> bool:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return any(c >= n for c in collections.Counter(lines).values())


def failure_kinds(preds: List[dict]) -> Dict[str, int]:
    c = collections.Counter()
    for p in preds:
        if p["correct"]:
            continue
        if is_loop(p["completion"]):
            c["loop"] += 1
        elif "####" not in p["completion"]:
            c["no_answer"] += 1
        else:
            c["wrong_answer"] += 1
    return dict(c)


def final_answer(text: str) -> str:
    if "####" not in text:
        return "NONE"
    return text.rsplit("####", 1)[1].strip().split("\n")[0].strip() or "NONE"


def eval_records(set_name: str) -> List[dict]:
    path = os.path.join(EVAL_DIR, set_name.split("@")[0] + ".jsonl")
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def by_field(preds: List[dict], recs: List[dict], field: str) -> Dict[object, Tuple[int, int]]:
    """
    {field value: (correct, total)}, joining predictions to records by prompt.
    """
    meta = {r["prompt"]: r.get(field) for r in recs}
    out: Dict[object, List[int]] = collections.defaultdict(lambda: [0, 0])
    for p in preds:
        v = meta.get(p["prompt"])
        if v is None:
            continue
        out[v][0] += int(p["correct"])
        out[v][1] += 1
    return {k: tuple(v) for k, v in sorted(out.items())}


def fmt(v) -> str:
    return "  -  " if v is None else f"{100 * v:5.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="+", help="run names under results/eval")
    ap.add_argument("--sft", nargs="*", default=[], help="SFT output dirs (per-epoch history)")
    a = ap.parse_args()
    metrics = {n: load_metrics(n) for n in a.names}
    sets = sorted({s for m in metrics.values() for s in m})
    print("## Accuracy (%)")
    print(f"{'set':28s} {'run':14s} " + " ".join(f"{c[:10]:>10s}" for c in COLS))
    for s in sets:
        for n in a.names:
            m = metrics[n].get(s)
            if m:
                print(f"{s:28s} {n:14s} " + " ".join(f"{fmt(m.get(c)):>10s}" for c in COLS))
    for n in a.names:
        preds = load_preds(n)
        print(f"\n## Failures of {n} (wrong greedy answers)")
        for s, (_, ps) in sorted(preds.items()):
            print(f"  {s:28s} wrong {sum(not p['correct'] for p in ps):5d} / {len(ps):5d}  {failure_kinds(ps)}")
        for s, (_, ps) in sorted(preds.items()):
            recs = eval_records(s)
            if "pw" in s or "proofwriter" in s:
                print(f"  {s}: predicted labels {dict(collections.Counter(final_answer(p['completion']) for p in ps).most_common(5))}")
                acc = {g: by_field([p for p in ps if p['gold'] == g], recs, 'answer') for g in ('yes', 'no', 'unknown')}
                print(f"  {s}: accuracy by gold label " + ", ".join(
                    f"{g} {100 * sum(c for c, _ in v.values()) / max(1, sum(t for _, t in v.values())):.0f}%" for g, v in acc.items()))
                if recs and "qdep" in recs[0]:
                    print(f"  {s}: accuracy by proof depth " + ", ".join(
                        f"d{k} {100 * c / t:.0f}%" for k, (c, t) in by_field(ps, recs, 'qdep').items()))
            elif recs and "n_ops" in recs[0]:
                print(f"  {s}: accuracy by steps " + ", ".join(
                    f"{k}: {100 * c / t:.0f}%" for k, (c, t) in by_field(ps, recs, 'n_ops').items()))
    for d in a.sft:
        path = os.path.join(d, "metrics.json")
        if not os.path.exists(path):
            print(f"\n## {d}: no metrics.json yet")
            continue
        hist = json.load(open(path))["per_epoch"]
        print(f"\n## SFT validation per epoch: {d}")
        keys = [k for k in hist[0] if k != "epoch"]
        print("  epoch " + " ".join(f"{k[:18]:>18s}" for k in keys))
        for h in hist:
            print(f"  {h['epoch']:5d} " + " ".join(f"{fmt(h[k]):>18s}" for k in keys))


if __name__ == "__main__":
    main()
