"""
Post-training data: a pool of training problems the SFT model has not seen,
sampling N completions per problem, and scoring them against the gold answer.

The scored samples feed both rejection-sampling fine-tuning (keep the correct
completions) and the verifier (correct / incorrect labels).

Problem pool (per source, in the same seeded order as SFT):
- real sources: records after the ones SFT used for validation and training;
- ProofWriter train rows after the ones SFT used;
- synthetic generators: new problems from a different seed.
Eval and validation prompts, and records overlapping the eval sets (13-grams),
are always excluded.

Import as:

import logic_slm_posttrain as lspost
"""

import glob
import gzip
import json
import random
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import logic_slm_format as lsform
import logic_slm_mixture as lsmix
import logic_slm_scorers as lssco
import logic_slm_sft as lssft


# #############################################################################
# Problem pool
# #############################################################################


def real_pool(
    spec: Dict[str, Tuple[str, int]], seed: int, held: Set[str], skip_val: int, scale: float,
    n_per_source: Dict[str, int], eval_index: Optional[Set[int]] = None,
    reuse: Optional[Set[str]] = None, extra_skip: int = 0,
) -> List[dict]:
    """
    Real problems not used by SFT: per source, walk the records in the same
    seeded order as `logic_slm_sft.load_real_pairs`, skip the first
    `skip_val + scale * max` usable ones (SFT validation + training), and
    take the next `n_per_source[name]`.

    :param reuse: sources whose problems are taken from the start even though
        SFT trained on them (e.g. GSM8K train, fully used by SFT); their
        records are tagged `seen_in_sft=True`
    :param extra_skip: further records to skip per (non-reused) source, e.g.
        the ones an earlier post-training stage already sampled
    """
    import logic_slm_decontam as lsdec

    out: List[dict] = []
    for name in sorted(spec):
        want = n_per_source.get(name, 0)
        if want <= 0:
            continue
        pattern, n_max = spec[name]
        seen_in_sft = name in (reuse or ())
        skip = 0 if seen_in_sft else skip_val + int(n_max * scale) + extra_skip
        paths = sorted(glob.glob(pattern))
        random.Random(lsmix.stable_seed(seed, "sft-real", name)).shuffle(paths)
        seen = got = 0
        for path in paths:
            with gzip.open(path, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    if r["prompt"] in held:
                        continue
                    if eval_index and lsdec.is_contaminated(r["prompt"] + r["solution"], eval_index):
                        continue
                    seen += 1
                    if seen <= skip:
                        continue
                    out.append({"prompt": r["prompt"], "answer": r["answer"],
                                "task": r.get("task", "math"), "gen": name,
                                "seen_in_sft": seen_in_sft})
                    got += 1
                    if got >= want:
                        break
            if got >= want:
                break
    return out


def proofwriter_pool(raw_dir: str, n: int, skip: int, seed: int, exclude_theories: Set[str]) -> List[dict]:
    """
    ProofWriter train problems after the first `skip` usable rows SFT used.
    """
    import logic_slm_proofwriter as lspw

    pairs, val, _ = lspw.load_sft_pairs(raw_dir, skip + n, 0, seed, exclude_theories)
    out = []
    for prompt, solution in pairs[skip:skip + n]:
        answer = solution.rsplit("####", 1)[-1].strip()
        out.append({"prompt": prompt, "answer": answer, "task": "logic", "gen": "proofwriter"})
    return out


def synthetic_pool(mixture: Dict[str, dict], n_per_unit_share: int, seed: int, held: Set[str],
                   overrides: Optional[Dict[str, dict]] = None, tag: str = "posttrain") -> List[dict]:
    """
    New synthetic problems (different seed from SFT) with their gold answers.
    """
    synth = {k: v for k, v in mixture.items() if not lsmix.is_stored(v)}
    for name, kw in (overrides or {}).items():
        synth[name] = {**synth[name], "train": {**synth[name]["train"], **kw}}
    out = []
    for name in sorted(synth):
        n = max(1, int(synth[name]["share"] * n_per_unit_share))
        got = 0
        for ex in lsmix.generate(synth, name, 3 * n, lsmix.stable_seed(seed, tag, name)):
            prompt = lsform.render_prompt(ex)
            if prompt in held:
                continue
            out.append({"prompt": prompt, "answer": ex.answer, "task": ex.task, "gen": name})
            got += 1
            if got == n:
                break
    return out


def filter_by_length(problems: Sequence[dict], tok, max_prompt_tokens: int) -> Tuple[List[dict], int]:
    """
    Drop problems whose prompt is longer than `max_prompt_tokens` (context
    length minus the room left for the answer); vLLM refuses longer prompts.

    :return: (kept problems, number dropped)
    """
    lens = tok([p["prompt"] for p in problems], add_special_tokens=False)["input_ids"]
    kept = [p for p, ids in zip(problems, lens) if len(ids) <= max_prompt_tokens]
    return kept, len(problems) - len(kept)


# #############################################################################
# Sampling
# #############################################################################


def sample_hf(model_dir: str, prompts: Sequence[str], n: int, temperature: float,
              max_new_tokens: int, batch_size: int = 128, seed: int = 0) -> List[List[str]]:
    """
    `n` sampled completions per prompt with Hugging Face `generate`.
    """
    import torch
    import transformers

    import logic_slm_eval as lseval

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = transformers.AutoTokenizer.from_pretrained(model_dir)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device)
    flat = [p for p in prompts for _ in range(n)]
    comps = lseval.generate_completions(model, tok, flat, max_new_tokens=max_new_tokens,
                                        batch_size=batch_size, do_sample=True,
                                        temperature=temperature)
    return [comps[i * n:(i + 1) * n] for i in range(len(prompts))]


def sample_vllm(model_dir: str, prompts: Sequence[str], n: int, temperature: float,
                max_new_tokens: int, seed: int = 0) -> List[List[str]]:
    """
    `n` sampled completions per prompt with vLLM.
    """
    import vllm

    llm = vllm.LLM(model=model_dir, dtype="bfloat16", seed=seed, enforce_eager=True,
                   gpu_memory_utilization=0.85, max_model_len=2048)
    params = vllm.SamplingParams(temperature=temperature, max_tokens=max_new_tokens, n=n, seed=seed)
    outs = llm.generate(list(prompts), params)
    return [[c.text for c in o.outputs] for o in outs]


def score_samples(problems: Sequence[dict], completions: Sequence[Sequence[str]]) -> List[dict]:
    """
    Attach completions and their correctness to each problem.
    """
    out = []
    for p, comps in zip(problems, completions):
        correct = [bool(lssco.exact_match(c, p["answer"], p["task"])) for c in comps]
        out.append({**p, "completions": list(comps), "correct": correct})
    return out


def summarize(samples: Sequence[dict]) -> Dict[str, float]:
    """
    Per-source and overall pass@1 / pass@n / share of problems with at least one correct sample.
    """
    by: Dict[str, List[dict]] = {}
    for s in samples:
        by.setdefault(s["gen"], []).append(s)
        by.setdefault("seen_in_sft" if s.get("seen_in_sft") else "unseen", []).append(s)
    res: Dict[str, float] = {}
    for g, ss in sorted(list(by.items()) + [("all", list(samples))]):
        n = len(ss[0]["correct"]) if ss else 0
        if not ss or n == 0:
            continue
        counts = [sum(s["correct"]) for s in ss]
        pk = lssco.pass_at_k_over_problems(counts, n, [1, n])
        res[f"{g}/pass@1"] = pk["pass@1"]
        res[f"{g}/pass@{n}"] = pk[f"pass@{n}"]
        res[f"{g}/n"] = len(ss)
    return res


# #############################################################################
# Rejection-sampling fine-tuning data
# #############################################################################


def rft_pairs(samples: Sequence[dict], max_per_problem: int, seed: int = 0) -> List[Tuple[str, str]]:
    """
    (prompt, completion) pairs from the correct samples: distinct completions
    only (whitespace-normalized), at most `max_per_problem` per problem.
    Completions without a final answer line are dropped.
    """
    rng = random.Random(seed)
    pairs: List[Tuple[str, str]] = []
    for s in samples:
        seen: Set[str] = set()
        good = []
        for c, ok in zip(s["completions"], s["correct"]):
            c = c.strip()
            key = " ".join(c.split())
            if ok and lsform.ANSWER_MARKER in c and key not in seen:
                seen.add(key)
                good.append(c)
        rng.shuffle(good)
        pairs.extend((s["prompt"], c) for c in good[:max_per_problem])
    rng.shuffle(pairs)
    return pairs


def write_jsonl(path: str, rows: Sequence[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            yield json.loads(line)
