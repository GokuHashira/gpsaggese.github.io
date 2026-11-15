#!/usr/bin/env python
"""
Pretrain a Llama-style model from scratch on a flat token file.

- Next-token loss on fixed-length blocks; the block order is a seeded
  permutation per epoch, so every model size sees the same data in the same
  order (Pythia-style) and resuming continues exactly where it stopped.
- AdamW (wd on matrices only), linear warmup + cosine decay to 10% of peak,
  bf16 autocast, gradient clipping, TensorBoard logging.
- At the end: saves the model + tokenizer and greedy-decodes the eval sets.

Example (CPU smoke test):

> python pretrain.py --size tiny --tokenizer models/tok --train-bin data/train.bin \
    --val-bin data/val.bin --out models/smoke --tokens 200000 --batch-tokens 8192 \
    --micro-batch 8 --seq-len 128
"""

import argparse
import json
import logging
import math
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

import logic_slm_data as lsdata
import logic_slm_eval as lseval
import logic_slm_model as lsmod
import logic_slm_tokenizer as lstok

_LOG = logging.getLogger("pretrain")


def lr_at(step: int, total: int, warmup: int, peak: float, min_ratio: float = 0.1) -> float:
    """
    Linear warmup, then cosine decay to `min_ratio * peak` at `total`.
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


class BlockStream:
    """
    Deterministic stream of (input, target) blocks from a token file.
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

    def _block(self, sample_idx: int) -> np.ndarray:
        epoch, pos = divmod(sample_idx, self.n_blocks)
        if epoch != self._perm_epoch:
            self._perm = np.random.default_rng(self.seed + epoch).permutation(self.n_blocks)
            self._perm_epoch = epoch
        start = int(self._perm[pos]) * self.seq_len
        return np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)

    def batch(self, first_sample: int, size: int) -> torch.Tensor:
        """
        Return a (size, seq_len + 1) tensor of samples first_sample.. .
        """
        return torch.from_numpy(np.stack([self._block(first_sample + i) for i in range(size)]))


def make_optimizer(model: torch.nn.Module, lr: float, wd: float) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=fused)


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
    ap.add_argument("--size", required=True, choices=sorted(lsmod.SIZES))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--val-bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=float, required=True, help="training tokens")
    ap.add_argument("--batch-tokens", type=int, default=524_288)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.0, help="peak LR (0 = size default)")
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-blocks", type=int, default=256)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--eval-jsonl", nargs="*", default=[], help="name=path pairs")
    ap.add_argument("--eval-limit", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--su-per-gpu-hour", type=float, default=48.0, help="for SU estimate")
    ap.add_argument("--compile", action="store_true")
    return ap.parse_args(argv)


def main(argv: List[str] = None) -> Dict[str, float]:
    a = parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    tok = lstok.load_tokenizer(a.tokenizer)
    cfg = lsmod.make_config(
        a.size, len(tok), eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        max_position_embeddings=a.seq_len,
    )
    model = lsmod.make_model(cfg).to(device)
    params = lsmod.count_params(model)
    _LOG.info("size=%s params=%s device=%s bf16=%s", a.size, params, device, use_bf16)
    train_model = torch.compile(model) if a.compile else model

    seqs_per_step = a.batch_tokens // a.seq_len
    if seqs_per_step % a.micro_batch:
        raise ValueError("batch-tokens / seq-len must be a multiple of micro-batch")
    accum = seqs_per_step // a.micro_batch
    total_steps = max(1, int(a.tokens // (seqs_per_step * a.seq_len)))
    warmup = max(1, int(a.warmup_frac * total_steps))
    peak_lr = a.lr or lsmod.PEAK_LR[a.size]
    opt = make_optimizer(model, peak_lr, a.weight_decay)

    train = BlockStream(lsdata.load_bin(a.train_bin), a.seq_len, a.seed)
    val = BlockStream(lsdata.load_bin(a.val_bin), a.seq_len, seed=0)
    _LOG.info(
        "train_tokens=%d blocks=%d steps=%d accum=%d warmup=%d peak_lr=%g epochs=%.2f",
        len(train.tokens), train.n_blocks, total_steps, accum, warmup, peak_lr,
        total_steps * seqs_per_step / train.n_blocks,
    )

    ckpt_path = os.path.join(a.out, "last.pt")
    step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        _LOG.info("Resumed from step %d", step)

    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(os.path.join(a.out, "tb"))
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump({**vars(a), "params": params, "total_steps": total_steps}, f, indent=2)

    model.train()
    t_start = time.time()
    t_log, tok_log = time.time(), 0
    while step < total_steps:
        lr = lr_at(step, total_steps, warmup, peak_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for micro in range(accum):
            first = (step * accum + micro) * a.micro_batch
            b = train.batch(first, a.micro_batch).to(device, non_blocking=True)
            with amp:
                loss = lm_loss(train_model, b) / accum
            loss.backward()
            loss_acc += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tok_log += seqs_per_step * a.seq_len
        if step % a.log_every == 0 or step == total_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t_log
            tps = tok_log / dt
            _LOG.info("step %d/%d loss %.4f lr %.2e gnorm %.2f tok/s %.0f",
                      step, total_steps, loss_acc, lr, gnorm, tps)
            tb.add_scalar("train/loss", loss_acc, step)
            tb.add_scalar("train/lr", lr, step)
            tb.add_scalar("train/grad_norm", gnorm, step)
            tb.add_scalar("perf/tokens_per_s", tps, step)
            t_log, tok_log = time.time(), 0
        if step % a.val_every == 0 or step == total_steps:
            vl = val_loss(model, val, a.val_blocks, a.micro_batch, device, amp)
            _LOG.info("step %d val_loss %.4f", step, vl)
            tb.add_scalar("val/loss", vl, step)
        if step % a.save_every == 0 and step < total_steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt_path)

    train_time = time.time() - t_start
    model.save_pretrained(os.path.join(a.out, "final"))
    tok.save_pretrained(os.path.join(a.out, "final"))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    trained_tokens = total_steps * seqs_per_step * a.seq_len
    metrics: Dict[str, float] = {
        "val_loss": val_loss(model, val, a.val_blocks, a.micro_batch, device, amp),
        "train_time_s": train_time,
        "tokens_per_s": trained_tokens / max(train_time, 1e-9),
        "non_embedding_params": params["non_embedding"],
    }
    metrics["su_per_1b_tokens"] = 1e9 / metrics["tokens_per_s"] / 3600 * a.su_per_gpu_hour
    for spec in a.eval_jsonl:
        name, path = spec.split("=", 1)
        m = lseval.evaluate_jsonl(
            model, tok, path, limit=a.eval_limit, max_new_tokens=a.max_new_tokens,
            out_path=os.path.join(a.out, f"pred_{name}.jsonl"),
        )
        for k, v in m.items():
            metrics[f"{name}/{k}"] = v
        _LOG.info("eval %s: %s", name, {k: round(v, 4) for k, v in m.items()})
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _LOG.info("metrics: %s", json.dumps(metrics, indent=2))
    tb.close()
    return metrics


if __name__ == "__main__":
    main()
