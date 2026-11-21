"""
Build the Hugging Face model card (README.md) for a released checkpoint.

The evaluation table comes from `results/eval/<run>/*/eval_metrics.json`;
training cost comes from the run ledger `results/runs.jsonl` (only real runs:
records with SU > 0 whose name matches this model's runs).

Import as:

import deploy.model_card as dmc
"""

import glob
import json
import os
from typing import Dict, List, Optional

SIZES = {
    "q125m": ("~119M", "768 hidden, 18 layers, 12 heads (4 KV), FFN 2,176", "5B"),
    "q0.5b": ("~358M", "896 hidden, 24 layers, 14 heads (2 KV), FFN 4,864 (Qwen2.5-0.5B shape)", "8B"),
    "q1.5b": ("~1.31B", "1,536 hidden, 28 layers, 12 heads (2 KV), FFN 8,960 (Qwen2.5-1.5B shape)", "12B"),
}
STAGES = {
    "pretrain": "pretrained only (base model)",
    "sft": "pretrained, then supervised fine-tuning (SFT)",
    "sft-fixes": "pretrained, then SFT with targeted fixes (ProofWriter proofs, state-tracking arithmetic)",
    "rft": "SFT, then rejection-sampling fine-tuning (RFT)",
    "grpo": "SFT, then GRPO reinforcement learning",
}
# Eval set -> display name (in table order).
SETS = [
    ("real_gsm8k-test", "GSM8K (1,319)"),
    ("real_svamp", "SVAMP (300)"),
    ("real_math500@1024", "MATH-500 (500, 1,024 tokens)"),
    ("real_math500", "MATH-500 (500, 512 tokens)"),
    ("real_pw-depth5", "ProofWriter depth-5 (2,100)"),
    ("m1_test_id", "Synthetic arithmetic, 1-4 steps"),
    ("m1_test_extrap", "Synthetic arithmetic, 5-8 steps"),
    ("l3_test_id", "Synthetic English logic"),
]
DATA = [
    ("FineMath-4+", "HuggingFaceTB/finemath", "ODC-By 1.0"),
    ("TinyGSM", "TinyGSM/TinyGSM", "see dataset card"),
    ("OpenMathInstruct-2", "nvidia/OpenMathInstruct-2", "CC-BY-4.0"),
    ("MetaMathQA", "meta-math/MetaMathQA", "MIT"),
    ("Orca-Math", "microsoft/orca-math-word-problems-200k", "MIT"),
    ("GSM8K (train)", "openai/gsm8k", "MIT"),
    ("ProofWriter (train, SFT + fixes only)", "tasksource/proofwriter", "CC-BY-4.0"),
]


def load_eval(eval_run: str, root: str = "results/eval") -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for path in sorted(glob.glob(os.path.join(root, eval_run, "*", "eval_metrics.json"))):
        sub = os.path.basename(os.path.dirname(path))
        for s, m in json.load(open(path))["per_set"].items():
            out[f"{s}@1024" if sub == "math1024" else s] = m
    return out


def ledger_costs(size: str, path: str = "results/runs.jsonl") -> List[dict]:
    """
    Real training runs of this size (pretraining and fine-tuning), latest per kind/name.
    """
    if not os.path.exists(path):
        return []
    runs: Dict[tuple, dict] = {}
    for line in open(path):
        r = json.loads(line)
        if (r.get("su") or 0) <= 1 or not str(r.get("name", "")).endswith(size):
            continue
        if r.get("kind") in ("pretrain", "sft"):
            runs[(r["kind"], r["name"])] = r
    return list(runs.values())


def _pct(v: Optional[float]) -> str:
    return "–" if v is None else f"{100 * v:.1f}"


def eval_table(ev: Dict[str, Dict[str, float]]) -> str:
    rows = ["| Benchmark | Greedy | Majority@8 | Pass@8 |", "|---|---|---|---|"]
    shown = set()
    for key, name in SETS:
        m = ev.get(key)
        if not m:
            continue
        if key == "real_math500" and "real_math500@1024" in shown:
            continue
        shown.add(key)
        rows.append(f"| {name} | {_pct(m.get('exact_match'))} | {_pct(m.get('majority@8'))} | "
                    f"{_pct(m.get('pass@8'))} |")
    return "\n".join(rows) if len(rows) > 2 else "_Evaluation results not available yet._"


def cost_table(runs: List[dict]) -> str:
    if not runs:
        return "_Training cost not recorded._"
    rows = ["| Stage | GPU | GPU-hours | Zaratan SU |", "|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: r.get("slurm_start_time", "")):
        rows.append(f"| {r['kind']} ({r['name']}) | {r.get('gpu', '?')} x {r.get('world', 1)} | "
                    f"{r.get('gpu_hours', 0):.1f} | {r.get('su', 0):,.0f} |")
    return "\n".join(rows)


def build_card(*, size: str, stage: str, eval_run: str, license_id: str, repo_id: str,
               model_dir: str = "") -> str:
    params, shape, tokens = SIZES[size]
    ev = load_eval(eval_run) if eval_run else {}
    runs = ledger_costs(size)
    datasets = "\n".join(f"- {hf}" for _, hf, _ in DATA)
    data_rows = "\n".join(f"| {n} | `{hf}` | {lic} |" for n, hf, lic in DATA)
    return f"""---
license: {license_id}
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- math
- reasoning
- small-language-model
- from-scratch
datasets:
{datasets}
---

# {repo_id}

A **{params}-parameter** decoder-only language model for **math word problems and
logical reasoning**, trained **from scratch** for the MSML610 (Advanced Machine
Learning, University of Maryland, Fall 2026) class project "Small Language Models
for Math and Logic". This checkpoint is **{STAGES.get(stage, stage)}**.

- Architecture: Qwen2-style (RMSNorm, RoPE, SwiGLU, grouped-query attention, QKV
  bias, tied embeddings); {shape}
- Context: 2,048 tokens. Tokenizer: 32k BPE with single-digit numbers.
- Pretraining: {tokens} tokens, ~90% open math data + ~10% synthetic
  step-by-step arithmetic and logic problems.

## How to use

The model expects this exact prompt format and writes a worked solution that ends
with `#### <answer>`:

```python
import transformers

tok = transformers.AutoTokenizer.from_pretrained("{repo_id}")
model = transformers.AutoModelForCausalLM.from_pretrained("{repo_id}")
prompt = "Problem: A baker makes 24 muffins and sells 3/4 of them. How many are left?\\nSolution:\\n"
ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
out = model.generate(**ids, max_new_tokens=512, do_sample=False)
text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
print(text)                                   # steps, then "#### 6"
print(text.rsplit("####", 1)[-1].strip())    # final answer
```

Logic questions use `Premises: <facts and rules>\\nQuestion: <statement>?\\nSolution:\\n`
and are answered `#### yes`, `#### no` or `#### unknown`.

Requires `transformers >= 5.0`.

## Evaluation (accuracy, %)

Greedy decoding; Majority@8 and Pass@8 use 8 samples at temperature 0.7.

{eval_table(ev)}

## Training cost

Trained on the University of Maryland Zaratan cluster (NVIDIA H100 80GB).

{cost_table(runs)}

## Training data

All training data was 13-gram decontaminated against every evaluation set.

| Dataset | Hub id | License |
|---|---|---|
{data_rows}
| Synthetic arithmetic / logic | generated by the project (z3-checked logic labels) | — |

## Limitations

- Math and logic only; not a chat or general-purpose model.
- Weak on problems longer than those seen in training (e.g. 5–8-step arithmetic),
  and on competition-level math (MATH-500).
- Answers can be confidently wrong; check the steps.
- English only; 2,048-token context.

## Authors

Gokulakannan Sakthivel and Basil Varghese (University of Maryland).
"""
