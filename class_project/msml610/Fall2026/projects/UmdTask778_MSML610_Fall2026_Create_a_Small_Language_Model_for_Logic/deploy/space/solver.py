"""
Model loading, prompting and answer extraction for the demo (no UI code, so it
can be tested without Gradio).

Import as:

import solver
"""

import functools
import os
import re
import time
from typing import Dict, Tuple

import torch
import transformers

# Display name -> Hub repo id (override with the SLM_MODELS env var:
# "Name=user/repo;Name2=user/repo2").
DEFAULT_MODELS: Dict[str, str] = {
    "358M (default)": "REPLACE_USER/slm-math-358m",
    "119M (fastest)": "REPLACE_USER/slm-math-119m",
    "1.31B (best, slow on CPU)": "REPLACE_USER/slm-math-1.3b",
}
MAX_PROMPT_TOKENS = 1536


def models() -> Dict[str, str]:
    spec = os.environ.get("SLM_MODELS", "")
    if not spec:
        return dict(DEFAULT_MODELS)
    out = {}
    for item in spec.split(";"):
        if "=" in item:
            name, repo = item.split("=", 1)
            out[name.strip()] = repo.strip()
    return out


@functools.lru_cache(maxsize=3)
def load(repo: str) -> Tuple[transformers.PreTrainedTokenizerBase, transformers.PreTrainedModel]:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tok = transformers.AutoTokenizer.from_pretrained(repo)
    model = transformers.AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32)
    model.eval()
    return tok, model


def math_prompt(problem: str) -> str:
    return f"Problem: {' '.join(problem.split())}\nSolution:\n"


def logic_prompt(premises: str, question: str) -> str:
    q = " ".join(question.split()).rstrip("?").rstrip(".")
    return f"Premises: {' '.join(premises.split())}\nQuestion: {q}?\nSolution:\n"


def final_answer(text: str) -> str:
    m = re.search(r"####\s*(.+)", text)
    return m.group(1).strip() if m else "(no final answer)"


def solve(repo: str, prompt: str, *, max_new_tokens: int = 512, sample: bool = False,
          temperature: float = 0.7) -> Dict[str, object]:
    """
    Generate a worked solution; returns steps, final answer and speed.
    """
    tok, model = load(repo)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
    if ids["input_ids"].shape[1] > MAX_PROMPT_TOKENS:
        return {"steps": "", "answer": "(problem too long for the 2,048-token context)",
                "tokens": 0, "seconds": 0.0}
    t = time.time()
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=sample,
                             temperature=temperature if sample else None,
                             top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    new = gen[0, ids["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=True)
    steps = text.split("####")[0].strip()
    return {"steps": steps, "answer": final_answer(text), "tokens": int(new.shape[0]),
            "seconds": round(time.time() - t, 2)}
