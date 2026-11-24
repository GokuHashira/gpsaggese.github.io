#!/usr/bin/env python
"""
Check that vLLM serves our models correctly and measure its speed.

Greedy decoding on an eval set with vLLM; reports accuracy (same scorer as
evaluate.py), agreement with a previous Hugging Face `generate` prediction file
(answer by answer), and generated tokens per second. Also times 8-sample
generation (pass@8 / majority@8 workload).

Example:

> python deploy/vllm_check.py --model models/sft2_q125m/best --eval data/processed/corpus_v2/eval/real_gsm8k-test.jsonl \
    --hf-preds results/eval/sft2_q125m/real/pred_real_gsm8k-test.jsonl
"""

import argparse
import collections
import json
import os
import time

import logic_slm_scorers as lssco


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--hf-preds", default="")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import vllm

    recs = [json.loads(line) for line in open(a.eval)]
    t0 = time.time()
    llm = vllm.LLM(model=a.model, dtype="bfloat16", enforce_eager=True, seed=0,
                   gpu_memory_utilization=0.85, max_model_len=2048)
    load_s = time.time() - t0
    prompts = [r["prompt"] for r in recs]
    greedy = vllm.SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens)
    t = time.time()
    outs = llm.generate(prompts, greedy)
    g_s = time.time() - t
    comps = [o.outputs[0].text for o in outs]
    g_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    correct = [lssco.exact_match(c, r["answer"], r["task"]) for c, r in zip(comps, recs)]
    res = {"model": a.model, "n": len(recs), "load_s": round(load_s, 1),
           "greedy_acc": sum(correct) / len(recs), "greedy_s": round(g_s, 1),
           "greedy_tok_per_s": round(g_tok / g_s)}
    if a.hf_preds and os.path.exists(a.hf_preds):
        hf = {p["prompt"]: p for p in map(json.loads, open(a.hf_preds))}
        same_ans = same_ok = n = 0
        for c, r, ok in zip(comps, recs, correct):
            p = hf.get(r["prompt"])
            if p is None:
                continue
            n += 1
            same_ok += int(bool(p["correct"]) == bool(ok))
            same_ans += int(c.rsplit("####", 1)[-1].strip()[:40] == p["completion"].rsplit("####", 1)[-1].strip()[:40])
        res.update({"hf_compared": n, "hf_same_correctness": same_ok / max(n, 1),
                    "hf_same_final_answer": same_ans / max(n, 1),
                    "hf_acc": sum(hf[r["prompt"]]["correct"] for r in recs if r["prompt"] in hf) / max(n, 1)})
    if a.n_samples > 0:
        samp = vllm.SamplingParams(temperature=0.7, max_tokens=a.max_new_tokens, n=a.n_samples, seed=0)
        t = time.time()
        souts = llm.generate(prompts, samp)
        s_s = time.time() - t
        s_tok = sum(len(c.token_ids) for o in souts for c in o.outputs)
        counts, maj = [], 0
        for o, r in zip(souts, recs):
            texts = [c.text for c in o.outputs]
            counts.append(sum(lssco.exact_match(x, r["answer"], r["task"]) for x in texts))
            answers = collections.Counter(x.rsplit("####", 1)[-1].strip() for x in texts if "####" in x)
            if answers:
                top = answers.most_common(1)[0][0]
                maj += int(lssco.exact_match("#### " + top, r["answer"], r["task"]))
        res.update({"sample_s": round(s_s, 1), "sample_tok_per_s": round(s_tok / s_s),
                    f"majority@{a.n_samples}": maj / len(recs),
                    **lssco.pass_at_k_over_problems(counts, a.n_samples, [1, a.n_samples])})
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
