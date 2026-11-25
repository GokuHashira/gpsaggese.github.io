#!/usr/bin/env python
"""
Supervised fine-tuning (SFT) of a pretrained checkpoint.

- Data: (prompt, solution) pairs generated from the corpus mixture with SFT
  seeds; eval / validation prompts are excluded. Loss only on solution tokens.
- Attention dropout 0.1, AdamW, warmup + cosine, bf16 autocast; single GPU or torchrun.
- After every epoch: greedy exact match and pass@k on validation problems of
  every generator. Following Cobbe et al. (sample diversity collapses after a
  few epochs), the checkpoint kept as `best/` maximizes pass@`--select-k`, not
  pass@1. `last/` holds the final epoch.
- At the end the best checkpoint is evaluated greedily on all eval sets.

Example:

> python sft.py --init models/pt_125m/final --corpus data/processed/corpus_v1 \
    --out models/sft_125m --epochs 4 --wandb-project logic-slm
"""

import argparse
import glob
import json
import logging
import os
import random
import shutil
import sys
import time
from typing import Dict, List

import torch
import transformers

import logic_slm_data as lsdata
import logic_slm_decontam as lsdec
import logic_slm_eval as lseval
import logic_slm_metrics as lsmet
import logic_slm_mixture as lsmix
import logic_slm_sft as lssft
import logic_slm_tokenizer as lstok
import logic_slm_train as lstrain

_LOG = logging.getLogger("sft")


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--init", required=True, help="pretrained model dir (save_pretrained)")
    ap.add_argument("--corpus", required=True, help="dir from build_corpus.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=200_000,
                    help="synthetic SFT examples when no real data is used (v1 corpora)")
    ap.add_argument("--n-synth", type=int, default=50_000,
                    help="synthetic SFT examples alongside real data (v2 corpora)")
    ap.add_argument("--real-scale", type=float, default=1.0,
                    help="fraction of each real source's cap (logic_slm_sft.REAL_SFT); 0 = synthetic only")
    ap.add_argument("--real-val-per-source", type=int, default=100,
                    help="held-out real problems per source for per-epoch pass@k")
    ap.add_argument("--real-eval-dir", default="data/processed/real/eval",
                    help="real eval sets: real SFT examples overlapping them (13-grams) are dropped")
    ap.add_argument("--synth-override", default="",
                    help='JSON {generator: {train kwarg: value}} applied to the synthetic SFT '
                         'generators, e.g. \'{"m1": {"state_steps": true}}\'')
    ap.add_argument("--proofwriter-n", type=int, default=0,
                    help="ProofWriter train examples with worked proofs (0 = none)")
    ap.add_argument("--proofwriter-val", type=int, default=100,
                    help="held-out ProofWriter train problems for per-epoch pass@k")
    ap.add_argument("--raw-dir", default="data/raw", help="raw downloads (ProofWriter parquet)")
    ap.add_argument("--extra-pairs", nargs="*", default=[],
                    help="JSONL files of {prompt, solution} added to the training data "
                         "(e.g. rejection-sampling fine-tuning pairs from posttrain_sample.py)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after this many optimizer steps (0 = all epochs); the LR schedule "
                         "and the last validation follow the shortened run (e.g. the matched-compute "
                         "control E11)")
    ap.add_argument("--batch-size", type=int, default=128, help="sequences per step (global)")
    ap.add_argument("--micro-batch", type=int, default=64, help="per GPU")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--val-per-gen", type=int, default=100, help="problems per generator for pass@k")
    ap.add_argument("--n-samples", type=int, default=8, help="samples per problem for pass@k")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--select-k", type=int, default=8, help="pick checkpoint by pass@k")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--eval-limit", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--start-epoch", type=int, default=1,
                    help="resume: first epoch to train (--init holds the weights after the "
                         "previous epoch; data order and LR schedule continue as in the full run; "
                         "the optimizer state is not saved, so AdamW starts fresh)")
    ap.add_argument("--resume-history", default="",
                    help="resume: JSON list of the per-epoch validation records already done "
                         "(sets the best metric / epoch that later epochs must beat)")
    ap.add_argument("--rewarm-steps", type=int, default=100,
                    help="resume: linear LR ramp over this many steps (fresh AdamW moments)")
    ap.add_argument("--compile", action="store_true")
    lstrain.add_common_args(ap)
    return ap.parse_args(argv)


def _val_records(corpus: str, per_gen: int) -> List[dict]:
    recs = lsdata.read_records(os.path.join(corpus, "val.jsonl"))
    by: Dict[str, List[dict]] = {}
    for r in recs:
        by.setdefault(r.get("gen", r["task"]), []).append(r)
    return [r for g in sorted(by) for r in by[g][:per_gen]]


def _save(model, tok, path: str, dtype: str) -> None:
    """
    Save the model and tokenizer without modifying the model being trained.

    A bf16 save writes a bf16 copy of the weights: casting the live model to
    bf16 and back would permanently round its fp32 weights.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    if dtype == "bf16":
        model.save_pretrained(path, state_dict=lstrain.bf16_state_dict(model))
        # Record the stored dtype, as `model.to(bf16)` would have.
        cfg = transformers.AutoConfig.from_pretrained(path)
        cfg.dtype = torch.bfloat16
        cfg.save_pretrained(path)
    else:
        model.save_pretrained(path)
    tok.save_pretrained(path)


def main(argv: List[str] = None) -> Dict[str, float]:
    a = parse_args(argv)
    d = lstrain.setup_dist()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO if d.is_main else logging.WARNING,
                        format="%(asctime)s %(message)s")
    if d.is_main:
        os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    device = d.device
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16)

    tok = lstok.load_tokenizer(a.init)
    # Llama only has dropout on the attention probabilities; that is the
    # regularizer used for SFT (--dropout).
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.init, attention_dropout=a.dropout, dtype=torch.float32
    ).to(device)
    timer = lsmet.Timer()
    with open(os.path.join(a.corpus, "manifest.json")) as f:
        manifest = json.load(f)
    mixture = manifest["mixture"]
    held = lsmix.load_held_prompts(a.corpus)
    # Synthetic part: generator entries only (stored real sources are pretraining-only).
    synth = {k: v for k, v in mixture.items() if not lsmix.is_stored(v)}
    for name, kw in (json.loads(a.synth_override) if a.synth_override else {}).items():
        if name not in synth:
            raise ValueError(f"--synth-override: unknown generator {name!r}")
        synth[name] = {**synth[name], "train": {**synth[name]["train"], **kw}}
    use_real = a.real_scale > 0 and manifest.get("version") == "v2"
    n_synth = a.n_synth if use_real else a.n_examples
    synth_share = sum(s["share"] for s in synth.values())
    pairs = lssft.build_sft_examples(synth, int(n_synth / synth_share), a.seed, held)
    data_counts: Dict[str, int] = {"synthetic": len(pairs)}
    real_val: List[dict] = []
    if use_real:
        index = lsdec.load_eval_index(a.real_eval_dir)[0] if os.path.isdir(a.real_eval_dir) else None
        rpairs, real_val, rcounts = lssft.load_real_pairs(
            lssft.REAL_SFT, a.seed, held, a.real_val_per_source, a.real_scale, index)
        pairs += rpairs
        data_counts.update(rcounts)
    if a.proofwriter_n > 0:
        import logic_slm_proofwriter as lspw

        excl = set()
        for path in glob.glob(os.path.join(a.corpus, "eval", "real_*proofwriter*.jsonl")) + \
                glob.glob(os.path.join(a.corpus, "eval", "real_pw-*.jsonl")):
            excl |= lspw.eval_theories(lsdata.read_records(path))
        ppairs, pval, pcounts = lspw.load_sft_pairs(a.raw_dir, a.proofwriter_n, a.proofwriter_val,
                                                    a.seed, excl)
        pairs += ppairs
        real_val += pval
        data_counts.update(pcounts)
    for path in a.extra_pairs:
        extra = [(r["prompt"], r["solution"]) for r in lsdata.read_records(path)]
        pairs += extra
        data_counts[f"extra:{os.path.basename(os.path.dirname(path)) or path}"] = len(extra)
    if use_real or a.proofwriter_n > 0 or a.extra_pairs:
        random.Random(a.seed).shuffle(pairs)
    ds = lssft.SFTDataset(pairs, tok, a.max_len)
    ds_tokens = sum(len(ids) for ids, _ in ds.items)
    _LOG.info("sft examples: %s (kept %d <= %d tokens, %d tokens)", data_counts, len(ds), a.max_len, ds_tokens)

    if a.batch_size % (a.micro_batch * d.world):
        raise ValueError("batch-size must be a multiple of micro-batch * GPUs")
    accum = a.batch_size // (a.micro_batch * d.world)
    steps_per_epoch = len(ds) // a.batch_size
    if steps_per_epoch < 1:
        raise ValueError(f"Only {len(ds)} SFT examples for batch size {a.batch_size}")
    total_steps = steps_per_epoch * a.epochs
    if a.max_steps > 0:
        total_steps = min(total_steps, a.max_steps)
        a.epochs = -(-total_steps // steps_per_epoch)  # epochs needed to cover total_steps
    warmup = max(1, int(a.warmup_frac * total_steps))
    opt = lstrain.make_optimizer(model, a.lr, a.weight_decay)
    train_model = model
    if d.world > 1:
        ids = [d.local_rank] if device.type == "cuda" else None
        train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=ids)
    if a.compile:
        train_model = torch.compile(train_model, dynamic=True)

    val_recs = _val_records(a.corpus, a.val_per_gen) + real_val
    run_info = {**vars(a), "n_train": len(ds), "total_steps": total_steps, "world": d.world,
                "data_counts": data_counts, "train_tokens_per_epoch": ds_tokens,
                "gpu": lstrain.gpu_name(), "su_per_gpu_hour": lstrain.su_rate(a.su_per_gpu_hour)}
    mon = lsmet.GpuMonitor(a.out).start() if d.is_main else None
    wb = tb = None
    if d.is_main:
        wb = lstrain.init_wandb(a.wandb_project, a.wandb_name, a.out, run_info)
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(os.path.join(a.out, "tb"))
        with open(os.path.join(a.out, "args.json"), "w") as f:
            json.dump(run_info, f, indent=2)

    step, best, best_epoch = 0, -1.0, -1
    history = []
    if a.start_epoch > 1:
        step = (a.start_epoch - 1) * steps_per_epoch
        if a.resume_history:
            with open(a.resume_history) as f:
                history = json.load(f)
            key = f"pass@{a.select_k}"
            for h in history:
                v = h.get(key, h.get("greedy_exact_match", -1.0))
                if v > best:
                    best, best_epoch = v, h["epoch"]
        _LOG.info("resuming at epoch %d (step %d), best so far %.4f at epoch %d",
                  a.start_epoch, step, best, best_epoch)
    resume_step = step
    t_start = time.time()
    for epoch in range(a.start_epoch, a.epochs + 1):
        g = torch.Generator().manual_seed(a.seed + epoch)
        order = torch.randperm(len(ds), generator=g).tolist()
        t_epoch = time.time()
        model.train()
        t_log = time.time()
        for s in range(steps_per_epoch):
            if step >= total_steps:
                break
            lr = lstrain.lr_at(step, total_steps, warmup, a.lr)
            if a.start_epoch > 1 and step - resume_step < a.rewarm_steps:
                lr *= (step - resume_step + 1) / a.rewarm_steps
            for grp in opt.param_groups:
                grp["lr"] = lr
            loss_acc = 0.0
            base = s * a.batch_size
            for m in range(accum):
                first = base + (m * d.world + d.rank) * a.micro_batch
                batch = ds.collate([ds[i] for i in order[first : first + a.micro_batch]])
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                sync = m == accum - 1 or d.world == 1
                ctx = train_model.no_sync() if (not sync and hasattr(train_model, "no_sync")) else _null()
                with ctx:
                    with amp:
                        loss = lssft.masked_lm_loss(train_model, batch) / accum
                    loss.backward()
                loss_acc += loss.item()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0:
                loss_avg = d.mean(loss_acc)
                if d.is_main:
                    _LOG.info("epoch %d step %d/%d loss %.4f lr %.2e gnorm %.2f (%.1fs)",
                              epoch, step, total_steps, loss_avg, lr, gnorm, time.time() - t_log)
                    tb.add_scalar("sft/loss", loss_avg, step)
                    tb.add_scalar("sft/lr", lr, step)
                    tb.add_scalar("sft/grad_norm", gnorm, step)
                t_log = time.time()
        timer.add("train", time.time() - t_epoch)
        t_val = time.time()
        if d.is_main:
            # Save the epoch before evaluating it, so a crash in the
            # evaluation can never lose the training of the epoch.
            if not a.no_save:
                _save(model, tok, os.path.join(a.out, "last"), a.save_dtype)
            model.eval()
            with amp:
                ev = lssft.pass_at_k_eval(
                    model, tok, val_recs, n_samples=a.n_samples, ks=a.ks,
                    temperature=a.temperature, max_new_tokens=a.max_new_tokens,
                )
            ev["epoch"] = epoch
            history.append(ev)
            _LOG.info("epoch %d val: %s", epoch, {k: round(v, 4) for k, v in ev.items()})
            for k, v in ev.items():
                tb.add_scalar(f"sft_val/{k}", v, epoch)
            key = f"pass@{a.select_k}" if f"pass@{a.select_k}" in ev else "greedy_exact_match"
            if ev[key] > best:
                best, best_epoch = ev[key], epoch
                if not a.no_save:
                    _save(model, tok, os.path.join(a.out, "best"), a.save_dtype)
        d.barrier()
        timer.add("val", time.time() - t_val)

    metrics: Dict[str, float] = {}
    if d.is_main:
        train_time = time.time() - t_start
        rate = run_info["su_per_gpu_hour"] * d.world
        metrics = {"best_epoch": best_epoch, "best_select_metric": best,
                   "train_time_s": train_time, "su_used": train_time / 3600 * rate}
        final = model
        if not a.no_save and best_epoch > 0:
            final = transformers.AutoModelForCausalLM.from_pretrained(
                os.path.join(a.out, "best")).to(device)
        t_eval = time.time()
        evals = sorted(glob.glob(os.path.join(a.corpus, "eval", "*.jsonl")))
        for path in evals:
            name = os.path.basename(path)[:-6]
            m = lseval.evaluate_jsonl(final, tok, path, limit=a.eval_limit,
                                      max_new_tokens=a.max_new_tokens,
                                      out_path=os.path.join(a.out, f"pred_{name}.jsonl"))
            for k, v in m.items():
                metrics[f"{name}/{k}"] = v
                tb.add_scalar(f"eval_{name}/{k}", v, a.epochs)
            _LOG.info("eval %s: %s", name, {k: round(v, 4) for k, v in m.items()})
        timer.add("eval", time.time() - t_eval)
        with open(os.path.join(a.out, "metrics.json"), "w") as f:
            json.dump({"final": metrics, "per_epoch": history}, f, indent=2)
        t_train = timer.phases.get("train", 0.0)
        lsmet.RunLedger(a.ledger).append({
            "kind": "sft", "name": os.path.basename(os.path.normpath(a.out)), "out": a.out,
            "init": a.init, "gpu": run_info["gpu"], "world": d.world, "epochs": a.epochs,
            "data_counts": data_counts, "n_train": len(ds), "tokens_trained": ds_tokens * a.epochs,
            "tokens_per_s": ds_tokens * a.epochs / t_train if t_train > 0 else 0.0,
            "best_epoch": best_epoch, "best_select_metric": best, "per_epoch": history,
            "eval_exact_match": {k.split("/")[0]: v for k, v in metrics.items() if k.endswith("/exact_match")},
            "git_commit": lsmet.git_commit(os.path.dirname(os.path.abspath(__file__))),
            **lsmet.slurm_info(),
            **lsmet.cost_fields(timer.total(), timer.phases, d.world, run_info["su_per_gpu_hour"]),
            **(mon.stop() if mon else {}),
        })
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
