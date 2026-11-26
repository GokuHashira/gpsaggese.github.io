#!/usr/bin/env python
"""
Prepare a trained checkpoint for the Hugging Face Hub (runs anywhere; no network).

Copies the model into a release folder, cleans the tokenizer config for use
outside Zaratan (2,048-token limit, no local-only keys), sets generation
defaults (greedy, 512 new tokens, stop at EOS), writes a model card
(`README.md`) with the evaluation results and training cost, then reloads
the folder with `transformers` and solves one problem as a smoke test.

Example:

> python deploy/prepare_release.py --model models/sft2_q125m/best --out models/release/slm-math-119m-sft \
    --size q125m --stage sft --eval-run sft2_q125m --license <license>
"""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_card  # noqa: E402

CTX = 2048
_DROP_TOKENIZER_KEYS = ("is_local", "local_files_only")
_SKIP_FILES = ("training_args.bin",)


def copy_model(src: str, out: str) -> None:
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for name in os.listdir(src):
        path = os.path.join(src, name)
        # training_args.bin (written by the GRPO trainer) is a pickle with local paths: never released.
        if os.path.isfile(path) and name not in _SKIP_FILES:
            shutil.copy2(path, os.path.join(out, name))


def clean_tokenizer_config(out: str) -> None:
    path = os.path.join(out, "tokenizer_config.json")
    cfg = json.load(open(path))
    for k in _DROP_TOKENIZER_KEYS:
        cfg.pop(k, None)
    cfg["model_max_length"] = CTX
    json.dump(cfg, open(path, "w"), indent=2)


def set_generation_defaults(out: str, max_new_tokens: int) -> None:
    path = os.path.join(out, "generation_config.json")
    cfg = json.load(open(path)) if os.path.exists(path) else {}
    cfg.pop("_from_model_config", None)
    cfg.update({"do_sample": False, "max_new_tokens": max_new_tokens})
    json.dump(cfg, open(path, "w"), indent=2)


def smoke_test(out: str, problem: str) -> str:
    """
    Load the release folder exactly as a Hub user would and solve one problem.
    """
    import torch
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(out)
    model = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    prompt = f"Problem: {problem}\nSolution:\n"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=256, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main(argv: Optional[List[str]] = None) -> Dict[str, str]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="checkpoint dir (save_pretrained)")
    ap.add_argument("--out", required=True, help="release folder to create")
    ap.add_argument("--size", required=True, help="q1m | q10m | q50m | q125m | q0.5b | q1.5b")
    ap.add_argument("--stage", required=True, help="pretrain | sft | sft-fixes | rft | grpo ...")
    ap.add_argument("--eval-run", default="", help="results/eval/<run> with this model's evaluation")
    ap.add_argument("--license", required=True, help="model license id for the card (team decision)")
    ap.add_argument("--repo-id", default="", help="final Hub id, used in the card's code example")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--family", action="store_true",
                    help="title the card with the family display name and list the sibling models")
    a = ap.parse_args(argv)
    copy_model(a.model, a.out)
    clean_tokenizer_config(a.out)
    set_generation_defaults(a.out, a.max_new_tokens)
    card = model_card.build_card(size=a.size, stage=a.stage, eval_run=a.eval_run,
                                 license_id=a.license, repo_id=a.repo_id or os.path.basename(a.out),
                                 model_dir=a.out, family=a.family)
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(card)
    result = {"out": a.out}
    if not a.no_smoke:
        problem = ("Natalia sold clips to 48 of her friends in April, and then she sold half as "
                   "many clips in May. How many clips did Natalia sell altogether in April and May?")
        result["smoke_completion"] = smoke_test(a.out, problem)
        print("smoke test completion:\n" + result["smoke_completion"])
    print(f"release folder ready: {a.out}")
    return result


if __name__ == "__main__":
    main()
