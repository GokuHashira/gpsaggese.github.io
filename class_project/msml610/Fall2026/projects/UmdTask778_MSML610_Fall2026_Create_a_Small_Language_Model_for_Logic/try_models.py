#!/usr/bin/env python
"""
Try the trained models on your own problems.

Each non-empty line of the prompts file is one problem (lines starting with # are ignored):

- a math problem in plain text, e.g. `Tom has 3 boxes of 12 pencils. He gives away 7. How many are left?`
- an arithmetic expression, e.g. `(45 + 27) * 3 - 18 = ?`
- an equation, e.g. `Find x if 3*x + 7 = 31.`
- a logic problem as `Premises: <facts and rules>. | Question: <question>?`

> python try_models.py --prompts my_prompts.txt --models models/grpo2_q125m/final --out answers.md
"""

import argparse
import re
import time

import torch
import transformers


def build_prompt(line: str) -> str:
    line = " ".join(line.split())
    if line.lower().startswith("premises:") and "|" in line:
        prem, q = line.split("|", 1)
        q = re.sub(r"^\s*question:\s*", "", q.strip(), flags=re.I).rstrip("?.")
        return f"{prem.strip()}\nQuestion: {q}?\nSolution:\n"
    line = re.sub(r"^\s*problem:\s*", "", line, flags=re.I)
    return f"Problem: {line}\nSolution:\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="model folders")
    ap.add_argument("--out", required=True, help="markdown file with the answers")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args()
    lines = [l.strip() for l in open(a.prompts) if l.strip() and not l.lstrip().startswith("#")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = ["# Model answers", ""]
    answers = {i: [] for i in range(len(lines))}
    for path in a.models:
        tok = transformers.AutoTokenizer.from_pretrained(path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
        for i, line in enumerate(lines):
            ids = tok(build_prompt(line), return_tensors="pt", add_special_tokens=False).to(dev)
            n = ids["input_ids"].shape[1]
            if n > 2048 - a.max_new_tokens:
                answers[i].append((path, "(problem too long)", "", 0.0))
                continue
            t = time.time()
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            text = tok.decode(gen[0, n:], skip_special_tokens=True).strip()
            m = re.search(r"####\s*(.+)", text)
            answers[i].append((path, m.group(1).strip() if m else "(no final answer)", text, time.time() - t))
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    for i, line in enumerate(lines):
        out += [f"## {i + 1}. {line}", "", "| Model | Final answer | Seconds |", "|---|---|---|"]
        out += [f"| `{p}` | {ans} | {s:.1f} |" for p, ans, _, s in answers[i]]
        out.append("")
        for p, _, text, _ in answers[i]:
            out += [f"`{p}`:", "", "```", text, "```", ""]
    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {a.out}: {len(lines)} problems x {len(a.models)} models")


if __name__ == "__main__":
    main()
