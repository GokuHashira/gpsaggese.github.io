#!/usr/bin/env python
"""
GRPO (group relative policy optimization) on the SFT model with TRL.

For every training problem the model writes a group of answers; the reward is
grounded in the dataset (no model judges anything):
- 1.0 if the final answer matches the dataset's answer (same exact-match
  scorer as the evaluation), else 0.0;
- plus `--format-weight` (default 0.1) if the completion ends with a single
  final `#### <answer>` line.
Answers better than their group's average are made more likely.

The problems come from `logic_slm_posttrain` (problems SFT did not train on).
`--use-vllm` generates with vLLM inside the training process (GPU only; run
with the envs/vllm environment).

Example:

> python grpo.py --init models/sftfix_q125m/best --out models/grpo_q125m --steps 500 --use-vllm
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from typing import Dict, List

import logic_slm_format as lsform
import logic_slm_metrics as lsmet
import logic_slm_scorers as lssco
import logic_slm_train as lstrain

_LOG = logging.getLogger("grpo")


def correctness_reward(completions: List[str], answer: List[str], task: List[str], **_) -> List[float]:
    """
    1.0 if the completion's final answer matches the dataset answer, else 0.0.
    """
    return [1.0 if lssco.exact_match(c, a, t) else 0.0 for c, a, t in zip(completions, answer, task)]


def format_reward(completions: List[str], **_) -> List[float]:
    """
    1.0 if the completion ends with exactly one final answer line, else 0.0.
    """
    out = []
    for c in completions:
        lines = [ln for ln in c.strip().split("\n") if ln.strip()]
        ok = bool(lines) and lines[-1].startswith(lsform.ANSWER_MARKER) and c.count(lsform.ANSWER_MARKER) == 1
        out.append(1.0 if ok else 0.0)
    return out


def vllm_ready_copy(model_dir: str, tmp_root: str) -> str:
    """
    Copy of the model whose tokenizer class vLLM's transformers can load.
    """
    dst = os.path.join(tmp_root, "init_model")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(model_dir, dst)
    p = os.path.join(dst, "tokenizer_config.json")
    cfg = json.load(open(p))
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    for k in ("is_local", "local_files_only", "backend"):
        cfg.pop(k, None)
    cfg["model_max_length"] = 2048
    json.dump(cfg, open(p, "w"), indent=2)
    return dst


IS_RATIO_KEY = "sampling/importance_sampling_ratio/mean"


def sampler_mismatch(logs: Dict[str, float], step: int, min_ratio: float, check_step: int) -> bool:
    """
    True if the sampler (vLLM) and the trainer disagree about the sampled answers.

    TRL logs the mean importance-sampling ratio between the trainer's and the
    sampler's log-probabilities; it is ~1 when both score the answers the same
    way. A low value means the policy-gradient signal is scaled down and
    distorted (e.g. dropout active in the trainer but not in the sampler, which
    made six GRPO runs learn nothing on 2026-11-24).
    """
    return step >= check_step and IS_RATIO_KEY in logs and float(logs[IS_RATIO_KEY]) < min_ratio


def make_mismatch_guard(min_ratio: float, check_step: int):
    """
    Trainer callback that stops the run at the first log line showing a
    sampler / trainer mismatch (see `sampler_mismatch`).
    """
    import transformers

    class _Guard(transformers.TrainerCallback):
        tripped = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and self.tripped is None and sampler_mismatch(logs, state.global_step,
                                                                  min_ratio, check_step):
                self.tripped = float(logs[IS_RATIO_KEY])
                _LOG.error("STOPPING: importance-sampling ratio %.3f < %.2f at step %d: the trainer "
                           "and the sampler score answers differently", self.tripped, min_ratio,
                           state.global_step)
                control.should_training_stop = True

    return _Guard()


def load_problems(a: argparse.Namespace) -> List[dict]:
    if a.problems:
        rows = [json.loads(line) for line in open(a.problems)]
    else:
        import posttrain_sample

        ns = argparse.Namespace(
            corpus=a.corpus, raw_dir=a.raw_dir, real_eval_dir=a.real_eval_dir, seed=a.seed,
            sft_val_per_source=100, sft_real_scale=1.0, sft_proofwriter=60000,
            per_real=a.per_real, proofwriter=a.proofwriter, synth_unit=a.synth_unit,
            synth_override=a.synth_override, reuse=a.reuse, pool_skip=a.pool_skip,
            synth_tag=a.synth_tag)
        rows = posttrain_sample.build_pool(ns)
    return [{"prompt": r["prompt"], "answer": r["answer"], "task": r["task"], "gen": r["gen"]} for r in rows]


def main(argv: List[str] = None) -> Dict[str, float]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--init", required=True, help="SFT model dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--problems", default="", help="JSONL of {prompt, answer, task, gen}; default: build the pool")
    ap.add_argument("--corpus", default="data/processed/corpus_v2")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--real-eval-dir", default="data/processed/real/eval")
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--per-real", type=int, default=3000)
    ap.add_argument("--proofwriter", type=int, default=3000)
    ap.add_argument("--synth-unit", type=int, default=30000)
    ap.add_argument("--synth-override", default='{"m1": {"state_steps": true}}')
    ap.add_argument("--reuse", nargs="*", default=[], help="see posttrain_sample.py --reuse")
    ap.add_argument("--pool-skip", type=int, default=0,
                    help="skip records an earlier stage used (RFT sampled 3,000 per source)")
    ap.add_argument("--synth-tag", default="grpo", help="seed tag for new synthetic problems")
    ap.add_argument("--pool-offset", type=int, default=0,
                    help="skip this many pool problems (e.g. the ones RFT sampled)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=32, help="distinct problems per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=32, help="completions per device forward pass")
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--max-context", type=int, default=2048, help="model context length")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0, help="KL penalty to the initial model")
    ap.add_argument("--format-weight", type=float, default=0.1)
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--vllm-memory", type=float, default=0.3)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--min-is-ratio", type=float, default=0.7,
                    help="stop the run if TRL's mean importance-sampling ratio is below this "
                         "(sampler / trainer mismatch); only logged with --use-vllm")
    ap.add_argument("--is-check-step", type=int, default=10)
    ap.add_argument("--cpu", action="store_true", help="CPU test mode (no bf16)")
    ap.add_argument("--wandb-project", default="")
    ap.add_argument("--ledger", default="results/runs.jsonl")
    a = ap.parse_args(argv)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(a.out, exist_ok=True)

    import datasets
    import torch
    import transformers
    from trl import GRPOConfig, GRPOTrainer

    t0 = time.time()
    tmp_root = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    init = vllm_ready_copy(a.init, tmp_root) if a.use_vllm else a.init
    tok = transformers.AutoTokenizer.from_pretrained(init)
    tok.padding_side = "left"
    import logic_slm_posttrain as lspost

    problems, dropped = lspost.filter_by_length(load_problems(a)[a.pool_offset:], tok,
                                                a.max_context - a.max_completion_length)
    _LOG.info("GRPO problems: %d (offset %d, %d dropped as too long)", len(problems),
              a.pool_offset, dropped)
    ds = datasets.Dataset.from_list(problems).shuffle(seed=a.seed)
    gen_per_step = a.prompts_per_step * a.num_generations
    if gen_per_step % a.micro_batch:
        raise ValueError("prompts-per-step * num-generations must be a multiple of micro-batch")
    cfg = GRPOConfig(
        output_dir=os.path.join(a.out, "trainer"),
        max_steps=a.steps,
        learning_rate=a.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=min(20, max(1, a.steps // 20)),
        per_device_train_batch_size=a.micro_batch,
        gradient_accumulation_steps=gen_per_step // a.micro_batch,
        num_generations=a.num_generations,
        max_completion_length=a.max_completion_length,
        temperature=a.temperature,
        beta=a.beta,
        reward_weights=[1.0, a.format_weight],
        logging_steps=5,
        save_strategy="steps",
        save_steps=a.save_every,
        save_total_limit=1,
        save_only_model=True,
        bf16=not a.cpu,
        use_cpu=a.cpu,
        use_vllm=a.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=a.vllm_memory,
        seed=a.seed,
        report_to=["wandb"] if a.wandb_project else [],
        run_name=os.path.basename(os.path.normpath(a.out)),
    )
    if a.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", a.wandb_project)
        os.environ.setdefault("WANDB_MODE", "offline")
    # No dropout in GRPO: the SFT / RFT checkpoints carry attention_dropout=0.1 (SFT's regularizer),
    # but the answers are sampled without dropout (vLLM, eval mode), so the trainer must score them
    # without dropout too; otherwise its log-probabilities disagree with the sampler's.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        init, attention_dropout=0.0, dtype=torch.float32 if a.cpu else torch.bfloat16)
    if getattr(model.config, "attention_dropout", 0.0) != 0.0:
        raise ValueError("GRPO needs attention_dropout=0")
    guard = make_mismatch_guard(a.min_is_ratio, a.is_check_step)
    trainer = GRPOTrainer(model=model, reward_funcs=[correctness_reward, format_reward], args=cfg,
                          train_dataset=ds, processing_class=tok, callbacks=[guard])
    mon = lsmet.GpuMonitor(a.out).start()
    t_train = time.time()
    trainer.train()
    train_s = time.time() - t_train
    final = os.path.join(a.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    hist = [h for h in trainer.state.log_history if "reward" in h]
    with open(os.path.join(a.out, "reward_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    first = hist[:5]
    last = hist[-5:]
    mean = lambda hs, k: sum(h.get(k, 0.0) for h in hs) / max(1, len(hs))
    metrics = {
        "steps": a.steps, "train_time_s": train_s, "n_problems": len(problems),
        "reward_start": mean(first, "reward"), "reward_end": mean(last, "reward"),
        "correct_start": mean(first, "rewards/correctness_reward/mean"),
        "correct_end": mean(last, "rewards/correctness_reward/mean"),
        "is_ratio_start": mean(first, IS_RATIO_KEY), "is_ratio_end": mean(last, IS_RATIO_KEY),
        "steps_done": trainer.state.global_step,
        "aborted_is_ratio": guard.tripped,
    }
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    lsmet.RunLedger(a.ledger).append({
        "kind": "grpo", "name": os.path.basename(os.path.normpath(a.out)), "out": a.out,
        "init": a.init, "args": vars(a), **metrics, **lsmet.slurm_info(),
        "gpu": lstrain.gpu_name(),
        **lsmet.cost_fields(time.time() - t0, {"train": train_s}, 1, lstrain.su_rate()),
        **mon.stop(),
    })
    shutil.rmtree(tmp_root, ignore_errors=True)
    _LOG.info("GRPO done: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    if guard.tripped is not None:
        # Non-zero exit: the job fails, so jobs chained with afterok (evaluation) do not start.
        raise SystemExit(2)
    return metrics


if __name__ == "__main__":
    main()
