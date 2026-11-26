#!/usr/bin/env python
"""
Confidence intervals for the evaluation results, from the saved per-problem predictions.

For every model size and stage: greedy accuracy with a 95% bootstrap interval on each real
benchmark. For every pair of consecutive stages (and GRPO vs the matched-compute control): the paired
difference on the same problems with its 95% bootstrap interval, so "better" can be told from noise.

> python confidence_intervals.py --out results/analysis/confidence.json --markdown results/analysis/confidence.md
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

SIZES = [("1M", "q1m"), ("10M", "q10m"), ("50M", "q50m"), ("119M", "q125m"), ("358M", "q0.5b"), ("1.31B", "q1.5b")]
# Stage label -> run-name pattern ({s} = size key).
STAGES = [("pretrained", "pt2_{s}"), ("sft_fixes", "sftfix_{s}"), ("rft", "rft_{s}"), ("grpo", "grpo2_{s}"),
          ("e11", "e11_{s}")]
# The valid 1M GRPO run is the rerun at the lower learning rate.
RUN_OVERRIDES = {("grpo", "q1m"): "grpo2_q1m_lr2e-5"}
SETS = [("GSM8K", "real", "pred_real_gsm8k-test.jsonl"), ("SVAMP", "real", "pred_real_svamp.jsonl"),
        ("MATH-500", "math1024", "pred_real_math500.jsonl"), ("ProofWriter-d5", "pw-depth5", "pred_real_pw-depth5.jsonl")]
PAIRS = [("sft_fixes", "pretrained"), ("rft", "sft_fixes"), ("grpo", "rft"), ("grpo", "e11")]


def bootstrap_ci(correct: np.ndarray, n_boot: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    """
    Mean of a 0/1 vector with its 95% percentile-bootstrap interval, all in percent.
    """
    correct = np.asarray(correct, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    means = correct[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return 100 * correct.mean(), 100 * lo, 100 * hi


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    """
    Accuracy of `a` minus accuracy of `b` on the same problems (percent points), with its 95%
    bootstrap interval; problems are resampled jointly, so the pairing is kept.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"paired vectors differ in length: {len(a)} vs {len(b)}")
    return bootstrap_ci(a - b, n_boot, seed)


def run_name(stage: str, size: str) -> str:
    return RUN_OVERRIDES.get((stage, size), dict(STAGES)[stage].format(s=size))


def load_correct(eval_dir: str, stage: str, size: str, part: str, fname: str) -> Optional[np.ndarray]:
    path = os.path.join(eval_dir, run_name(stage, size), part, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return np.array([bool(json.loads(line)["correct"]) for line in f if line.strip()])


def compute(eval_dir: str, n_boot: int) -> Dict[str, dict]:
    out: Dict[str, dict] = {"accuracy": {}, "paired_difference": {}}
    for label, size in SIZES:
        for set_name, part, fname in SETS:
            vecs = {st: load_correct(eval_dir, st, size, part, fname) for st, _ in STAGES}
            for st, v in vecs.items():
                if v is not None:
                    m, lo, hi = bootstrap_ci(v, n_boot)
                    out["accuracy"].setdefault(set_name, {}).setdefault(label, {})[st] = {
                        "n": int(len(v)), "acc": round(m, 2), "lo": round(lo, 2), "hi": round(hi, 2)}
            for a, b in PAIRS:
                if vecs[a] is not None and vecs[b] is not None and len(vecs[a]) == len(vecs[b]):
                    m, lo, hi = paired_diff_ci(vecs[a], vecs[b], n_boot)
                    out["paired_difference"].setdefault(set_name, {}).setdefault(label, {})[f"{a}-{b}"] = {
                        "diff": round(m, 2), "lo": round(lo, 2), "hi": round(hi, 2),
                        "significant": bool(lo > 0 or hi < 0)}
    return out


def to_markdown(res: Dict[str, dict]) -> str:
    lines: List[str] = ["# Confidence intervals (95%, bootstrap over problems)", ""]
    stages = [st for st, _ in STAGES]
    for set_name, by_size in res["accuracy"].items():
        lines += [f"## {set_name}: greedy accuracy, % [95% interval]", "",
                  "| Size | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
        for label, row in by_size.items():
            cells = [f"{row[s]['acc']:.1f} [{row[s]['lo']:.1f}, {row[s]['hi']:.1f}]" if s in row else "–" for s in stages]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
        pairs = [f"{a}-{b}" for a, b in PAIRS]
        lines += [f"## {set_name}: paired difference, points [95% interval]; * = interval excludes 0", "",
                  "| Size | " + " | ".join(pairs) + " |", "|---|" + "---|" * len(pairs)]
        for label, row in res["paired_difference"].get(set_name, {}).items():
            cells = [(f"{row[p]['diff']:+.1f} [{row[p]['lo']:+.1f}, {row[p]['hi']:+.1f}]" + ("*" if row[p]["significant"] else ""))
                     if p in row else "–" for p in pairs]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="results/eval")
    ap.add_argument("--out", default="results/analysis/confidence.json")
    ap.add_argument("--markdown", default="results/analysis/confidence.md")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    res = compute(a.eval_dir, a.n_boot)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    with open(a.markdown, "w") as f:
        f.write(to_markdown(res) + "\n")
    print(f"wrote {a.out} and {a.markdown}")


if __name__ == "__main__":
    main()
