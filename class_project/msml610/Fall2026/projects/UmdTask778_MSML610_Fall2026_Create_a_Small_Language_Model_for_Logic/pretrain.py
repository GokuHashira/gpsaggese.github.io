#!/usr/bin/env python
"""
Pretrain a Llama-style model from scratch.

Training data comes either from a corpus directory built by `build_corpus.py`
(blocks are generated on the fly by `logic_slm_mixture.MixtureBlocks`, so no
corpus is stored) or from a fixed token file (`--train-bin`).

- Step `s` always uses blocks `s * B .. (s + 1) * B - 1` (B = sequences per
  step), so every model size sees the same data in the same order
  (Pythia-style), independently of the number of GPUs, and resuming continues
  exactly where it stopped.
- AdamW (wd on matrices only), linear warmup + cosine decay to 10% of peak,
  bf16 autocast, gradient clipping, TensorBoard (+ optional W&B) logging.
- Single GPU, or several GPUs with `torchrun --nproc_per_node=N pretrain.py ...`.
- At the end: saves the model + tokenizer and greedy-decodes the eval sets.

Examples:

> python pretrain.py --size 125m --corpus data/processed/corpus_v1 \
    --out models/pt_125m --tokens 2.5e9 --wandb-project logic-slm
> python pretrain.py --size tiny --tokenizer models/tok --train-bin data/train.bin \
    --val-bin data/val.bin --out models/smoke --tokens 200000 --batch-tokens 8192 \
    --micro-batch 8 --seq-len 128
"""

import argparse
import glob
import json
import logging
import sys
import os
import time
from typing import Dict, Iterator, List

import numpy as np
import torch
import torch.nn.functional as F
import transformers

import logic_slm_data as lsdata
import logic_slm_eval as lseval
import logic_slm_metrics as lsmet
import logic_slm_mixture as lsmix
import logic_slm_model as lsmod
import logic_slm_tokenizer as lstok
import logic_slm_train as lstrain

_LOG = logging.getLogger("pretrain")

# Re-exported for tests and older callers.
lr_at = lstrain.lr_at


class BlockStream(torch.utils.data.Dataset):
    """
    Blocks of `seq_len + 1` tokens from a token file, in a seeded order that
    is a fresh permutation every epoch.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, seed: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_blocks = (len(tokens) - 1) // seq_len
        if self.n_blocks < 1:
            raise ValueError("Token file shorter than one block")
        self.seed = seed
        self._perm_epoch = -1
        self._perm = None

    def __len__(self) -> int:
        return 10**12  # sample indices wrap around epochs

    def __getitem__(self, sample_idx: int) -> torch.Tensor:
        epoch, pos = divmod(sample_idx, self.n_blocks)
        if epoch != self._perm_epoch:
            self._perm = np.random.default_rng(self.seed + epoch).permutation(self.n_blocks)
            self._perm_epoch = epoch
        start = int(self._perm[pos]) * self.seq_len
        return torch.from_numpy(
            np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        )

    def batch(self, first_sample: int, size: int) -> torch.Tensor:
        """
        Return a (size, seq_len + 1) tensor of samples first_sample.. .
        """
        return torch.stack([self[first_sample + i] for i in range(size)])


class StepBatches(torch.utils.data.Sampler):
    """
    Yields this rank's micro-batch index lists for steps `start .. total - 1`.

    All ranks together cover blocks `s * seqs_per_step .. (s + 1) * seqs_per_step - 1`
    at step `s`, whatever the world size.
    """

    def __init__(self, start: int, total: int, seqs_per_step: int, micro: int, rank: int, world: int):
        self.start, self.total = start, total
        self.seqs, self.micro, self.rank, self.world = seqs_per_step, micro, rank, world
        self.accum = seqs_per_step // (micro * world)

    def __iter__(self) -> Iterator[List[int]]:
        for s in range(self.start, self.total):
            base = s * self.seqs
            for m in range(self.accum):
                first = base + (m * self.world + self.rank) * self.micro
                yield list(range(first, first + self.micro))

    def __len__(self) -> int:
        return (self.total - self.start) * self.accum


def lm_loss(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    x, y = batch[:, :-1], batch[:, 1:]
    logits = model(input_ids=x).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def val_loss(model, stream: BlockStream, n_blocks: int, micro: int, device, amp) -> float:
    model.eval()
    losses = []
    n = min(n_blocks, stream.n_blocks)
    for i in range(0, n, micro):
        b = stream.batch(i, min(micro, n - i)).to(device)
        with amp:
            losses.append(lm_loss(model, b).item())
    model.train()
    return float(np.mean(losses))


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", required=True, choices=lsmod.ALL_SIZES)
    ap.add_argument("--corpus", default="", help="dir from build_corpus.py")
    ap.add_argument("--tokenizer", default="", help="default: <corpus>/tokenizer")
    ap.add_argument("--train-bin", default="", help="fixed token file instead of streaming")
    ap.add_argument("--val-bin", default="", help="default: <corpus>/val.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=float, required=True, help="training tokens")
    ap.add_argument("--batch-tokens", type=int, default=524_288)
    ap.add_argument("--micro-batch", type=int, default=32, help="per GPU")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.0, help="peak LR (0 = size default)")
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--data-workers", type=int, default=8, help="per GPU, for streaming")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-blocks", type=int, default=256)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--eval-jsonl", nargs="*", default=None,
                    help="name=path pairs (default: all <corpus>/eval/*.jsonl)")
    ap.add_argument("--eval-limit", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--deadline-unix", type=float, default=0.0,
                    help="save a checkpoint and stop (marker INCOMPLETE) after this time")
    ap.add_argument("--rewarm-steps", type=int, default=50,
                    help="LR ramp after resuming from a weights-only checkpoint")
    ap.add_argument("--drop-sources", default="",
                    help="comma-separated mixture entries to remove (e.g. l1,l2,l3 for the "
                         "no-logic ablation E10); the other shares are renormalized")
    ap.add_argument("--curriculum", action="store_true",
                    help="easy-to-hard curriculum over the training tokens (experiment E9)")
    lstrain.add_common_args(ap)
    a = ap.parse_args(argv)
    if a.corpus:
        a.tokenizer = a.tokenizer or os.path.join(a.corpus, "tokenizer")
        a.val_bin = a.val_bin or os.path.join(a.corpus, "val.bin")
        if a.eval_jsonl is None:
            a.eval_jsonl = [
                f"{os.path.basename(p)[:-6]}={p}"
                for p in sorted(glob.glob(os.path.join(a.corpus, "eval", "*.jsonl")))
            ]
    if not (a.tokenizer and a.val_bin and (a.train_bin or a.corpus)):
        ap.error("need --corpus, or --tokenizer + --val-bin + --train-bin")
    a.eval_jsonl = a.eval_jsonl or []
    return a


def _train_dataset(a: argparse.Namespace) -> torch.utils.data.Dataset:
    if a.train_bin:
        return BlockStream(lsdata.load_bin(a.train_bin), a.seq_len, a.seed)
    with open(os.path.join(a.corpus, "manifest.json")) as f:
        mixture = json.load(f)["mixture"]
    drop = {s for s in a.drop_sources.split(",") if s}
    unknown = drop - set(mixture)
    if unknown:
        raise ValueError(f"--drop-sources: unknown entries {sorted(unknown)}")
    mixture = {k: v for k, v in mixture.items() if k not in drop}
    return lsmix.MixtureBlocks(
        mixture, a.tokenizer, a.seq_len, a.seed,
        held_prompts=lsmix.load_held_prompts(a.corpus), corpus_dir=a.corpus,
        curriculum_blocks=int(a.tokens // a.seq_len) if a.curriculum else 0,
    )


def _per_rank_compile_caches() -> None:
    """
    Give each torchrun process its own compile cache dir, so concurrent
    processes never read each other's half-written kernel files.
    """
    rank = os.environ.get("LOCAL_RANK")
    if rank is None:
        return
    for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        if key in os.environ and not os.environ[key].endswith(f"rank{rank}"):
            os.environ[key] = os.path.join(os.environ[key], f"rank{rank}")


def main(argv: List[str] = None) -> Dict[str, float]:
    a = parse_args(argv)
    timer = lsmet.Timer()
    _per_rank_compile_caches()
    d = lstrain.setup_dist()
    logging.basicConfig(stream=sys.stdout, 
        level=logging.INFO if d.is_main else logging.WARNING,
        format="%(asctime)s %(message)s",
    )
    if d.is_main:
        os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    device = d.device
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok = lstok.load_tokenizer(a.tokenizer)
    cfg = lsmod.make_config(
        a.size, len(tok), eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        max_position_embeddings=a.seq_len,
    )
    model = lsmod.make_model(cfg).to(device)
    params = lsmod.count_params(model)
    _LOG.info("size=%s params=%s device=%s world=%d bf16=%s", a.size, params, device, d.world, use_bf16)

    seqs_per_step = a.batch_tokens // a.seq_len
    if seqs_per_step % (a.micro_batch * d.world):
        raise ValueError("batch-tokens / seq-len must be a multiple of micro-batch * GPUs")
    accum = seqs_per_step // (a.micro_batch * d.world)
    total_steps = max(1, int(a.tokens // (seqs_per_step * a.seq_len)))
    warmup = max(1, int(a.warmup_frac * total_steps))
    peak_lr = a.lr or lsmod.PEAK_LR[a.size]
    opt = lstrain.make_optimizer(model, peak_lr, a.weight_decay)

    ckpt_path = os.path.join(a.out, "last.pt")
    final_dir = os.path.join(a.out, "final")
    marker = os.path.join(a.out, "INCOMPLETE")
    # `prior_time` is the training wall time of earlier jobs of this run.
    step, rewarm_from, prior_time = 0, -1, 0.0
    eval_only = os.path.exists(os.path.join(final_dir, "config.json"))
    if eval_only:
        # Training already finished (e.g. the job stopped during evaluation).
        model = transformers.AutoModelForCausalLM.from_pretrained(final_dir).to(device)
        step = total_steps
        _LOG.info("Found %s: skipping training, evaluating only", final_dir)
    elif os.path.exists(ckpt_path):
        step, prior_time = lstrain.load_checkpoint(ckpt_path, model, opt, device)
        if not opt.state:
            rewarm_from = step  # fresh optimizer moments: ramp the LR back up
        _LOG.info("Resumed from step %d (optimizer state %s)", step,
                  "restored" if opt.state else "reset, re-warming LR")

    train_model = model
    if d.world > 1:
        ids = [d.local_rank] if device.type == "cuda" else None
        train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=ids)
    if a.compile:
        train_model = torch.compile(train_model)

    train_ds = _train_dataset(a)
    val = BlockStream(lsdata.load_bin(a.val_bin), a.seq_len, seed=0)
    streaming = not a.train_bin
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_sampler=StepBatches(step, total_steps, seqs_per_step, a.micro_batch, d.rank, d.world),
        collate_fn=torch.stack,
        num_workers=a.data_workers if streaming else 0,
        pin_memory=device.type == "cuda",
        prefetch_factor=4 if streaming and a.data_workers > 0 else None,
        persistent_workers=False,
    )
    _LOG.info(
        "data=%s steps=%d (from %d) seqs/step=%d accum=%d warmup=%d peak_lr=%g",
        "stream:" + a.corpus if streaming else a.train_bin, total_steps, step,
        seqs_per_step, accum, warmup, peak_lr,
    )

    run_info = {**vars(a), "params": params, "total_steps": total_steps, "world": d.world,
                "gpu": lstrain.gpu_name(), "su_per_gpu_hour": lstrain.su_rate(a.su_per_gpu_hour)}
    wb = tb = None
    if d.is_main:
        wb = lstrain.init_wandb(a.wandb_project, a.wandb_name, a.out, run_info, resume=True)
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(os.path.join(a.out, "tb"))
        with open(os.path.join(a.out, "args.json"), "w") as f:
            json.dump(run_info, f, indent=2)
    mon = lsmet.GpuMonitor(a.out).start() if d.is_main else None
    start_step = step
    fpt = lsmet.flops_per_token(params["non_embedding"], cfg.num_hidden_layers, cfg.hidden_size,
                                a.seq_len, len(tok))

    def ledger(kind: str, extra: Dict) -> None:
        """
        Append this run's metrics to the run ledger (main process only).
        """
        if not d.is_main:
            return
        run_tokens = (step - start_step) * seqs_per_step * a.seq_len
        t_train = timer.phases.get("train", 0.0)
        tps = run_tokens / t_train if t_train > 0 else 0.0
        rec = {
            "kind": kind, "name": os.path.basename(os.path.normpath(a.out)), "out": a.out,
            "size": a.size, "arch": getattr(cfg, "model_type", "llama"),
            "params_total": params["total"], "params_non_embedding": params["non_embedding"],
            "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "vocab": len(tok),
            "seq_len": a.seq_len, "gpu": run_info["gpu"], "world": d.world,
            "steps_total": total_steps, "step_start": start_step, "step_end": step,
            "tokens_target": total_steps * seqs_per_step * a.seq_len, "tokens_trained": run_tokens,
            "tokens_per_s": tps, "flops_per_token": fpt,
            "mfu": lsmet.mfu(tps, fpt, d.world, run_info["gpu"]),
            "corpus": a.corpus or a.train_bin, "seed": a.seed, "peak_lr": peak_lr,
            "batch_tokens": a.batch_tokens, "git_commit": lsmet.git_commit(os.path.dirname(os.path.abspath(__file__))),
            **lsmet.slurm_info(),
            **lsmet.cost_fields(timer.total(), timer.phases, d.world, run_info["su_per_gpu_hour"]),
            **(mon.stop() if mon else {}),
            **extra,
        }
        lsmet.RunLedger(a.ledger).append(rec)

    model.train()
    start_step = step
    t_start = time.time()
    t_log, tok_log = time.time(), 0
    batches = iter(loader)
    while step < total_steps:
        lr = lstrain.lr_at(step, total_steps, warmup, peak_lr)
        if rewarm_from >= 0 and step - rewarm_from < a.rewarm_steps:
            lr *= (step - rewarm_from + 1) / a.rewarm_steps
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for micro in range(accum):
            b = next(batches).to(device, non_blocking=True)
            sync = micro == accum - 1 or d.world == 1
            ctx = train_model.no_sync() if (not sync and hasattr(train_model, "no_sync")) else _null()
            with ctx:
                with amp:
                    loss = lm_loss(train_model, b) / accum
                loss.backward()
            loss_acc += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step == start_step + 1:
            # First step includes torch.compile and data-loader warm-up.
            if device.type == "cuda":
                torch.cuda.synchronize()
            timer.add("first_step", time.time() - t_start)
        tok_log += seqs_per_step * a.seq_len
        if step % a.log_every == 0 or step == total_steps:
            loss_avg = d.mean(loss_acc)
            if device.type == "cuda":
                torch.cuda.synchronize()
            tps = tok_log / (time.time() - t_log)
            if d.is_main:
                _LOG.info("step %d/%d loss %.4f lr %.2e gnorm %.2f tok/s %.0f",
                          step, total_steps, loss_avg, lr, gnorm, tps)
                tb.add_scalar("train/loss", loss_avg, step)
                tb.add_scalar("train/lr", lr, step)
                tb.add_scalar("train/grad_norm", gnorm, step)
                tb.add_scalar("perf/tokens_per_s", tps, step)
                if device.type == "cuda":
                    tb.add_scalar("perf/max_mem_gb", torch.cuda.max_memory_allocated(device) / 2**30, step)
            t_log, tok_log = time.time(), 0
        if step % a.val_every == 0 or step == total_steps:
            if d.is_main:
                vl = val_loss(model, val, a.val_blocks, a.micro_batch, device, amp)
                _LOG.info("step %d val_loss %.4f", step, vl)
                tb.add_scalar("val/loss", vl, step)
            d.barrier()
        if step % a.save_every == 0 and step < total_steps and not a.no_save:
            if d.is_main:
                lstrain.save_checkpoint(ckpt_path, model, opt, step, a.ckpt_weights_only,
                                        prior_time + time.time() - t_start)
            d.barrier()
        if a.deadline_unix and step < total_steps:
            # Rank 0 decides so all ranks stop at the same step.
            late = d.mean(float(d.is_main and time.time() > a.deadline_unix)) > 0
            if late:
                if d.is_main:
                    if not a.no_save:
                        lstrain.save_checkpoint(ckpt_path, model, opt, step, a.ckpt_weights_only,
                                                prior_time + time.time() - t_start)
                    with open(marker, "w") as f:
                        f.write(f"stopped at step {step}/{total_steps} before the time limit\n")
                    _LOG.info("Deadline reached: checkpoint at step %d, resubmit to continue", step)
                    timer.add("train", time.time() - t_start)
                    ledger("pretrain_partial", {"stopped_reason": "deadline"})
                    tb.close()
                    if wb is not None:
                        wb.finish()
                d.barrier()
                lstrain.cleanup_dist(d)
                return {"stopped_at_step": step}

    # Throughput uses only the steps of this job; the SU cost covers all jobs.
    job_time = time.time() - t_start
    timer.add("train", job_time)
    metrics: Dict[str, float] = {}
    if d.is_main:
        job_tokens = (total_steps - start_step) * seqs_per_step * a.seq_len
        rate = run_info["su_per_gpu_hour"] * d.world
        metrics = {
            "val_loss": val_loss(model.float(), val, a.val_blocks, a.micro_batch, device, amp),
            "train_time_s": prior_time + job_time,
            "non_embedding_params": params["non_embedding"],
            "gpus": d.world,
        }
        if job_tokens > 0:
            metrics["tokens_per_s"] = job_tokens / max(job_time, 1e-9)
            metrics["su_per_1b_tokens"] = 1e9 / metrics["tokens_per_s"] / 3600 * rate
        metrics["su_used_train"] = metrics["train_time_s"] / 3600 * rate
        if a.save_dtype == "bf16":
            model = model.to(torch.bfloat16)
        if eval_only:
            prev = os.path.join(a.out, "metrics.json")
            if os.path.exists(prev):
                with open(prev) as f:
                    old = json.load(f)
                for k in ("train_time_s", "tokens_per_s", "su_per_1b_tokens", "su_used_train"):
                    if k in old:
                        metrics[k] = old[k]
        elif not a.no_save:
            model.save_pretrained(final_dir)
            tok.save_pretrained(final_dir)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            # Partial metrics so an eval-only rerun can keep the training stats.
            with open(os.path.join(a.out, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        t_eval = time.time()
        for spec in a.eval_jsonl:
            name, path = spec.split("=", 1)
            m = lseval.evaluate_jsonl(
                model, tok, path, limit=a.eval_limit, max_new_tokens=a.max_new_tokens,
                out_path=os.path.join(a.out, f"pred_{name}.jsonl"),
            )
            for k, v in m.items():
                metrics[f"{name}/{k}"] = v
                tb.add_scalar(f"eval_{name}/{k}", v, total_steps)
            _LOG.info("eval %s: %s", name, {k: round(v, 4) for k, v in m.items()})
        timer.add("eval", time.time() - t_eval)
        with open(os.path.join(a.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        ledger("pretrain_eval_only" if eval_only else "pretrain", {
            "val_loss": metrics["val_loss"],
            "eval_exact_match": {k.split("/")[0]: v for k, v in metrics.items() if k.endswith("/exact_match")},
        })
        if os.path.exists(marker):
            os.remove(marker)
        _LOG.info("metrics: %s", json.dumps(metrics, indent=2))
        tb.close()
        if wb is not None:
            wb.summary.update(metrics)
            wb.finish()
    d.barrier()
    lstrain.cleanup_dist(d)
    return metrics


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
