"""
Training utilities shared by pretraining and fine-tuning scripts.

- LR schedule, AdamW parameter groups
- Zaratan SU accounting from the GPU name
- optional W&B run mirroring TensorBoard (offline on compute nodes)
- single-process or DDP (torchrun) setup
- compact checkpoints (weights-only / bf16) for the home-directory quota

Import as:

import logic_slm_train as lstrain
"""

import argparse
import dataclasses
import datetime
import logging
import math
import os
import uuid
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

_LOG = logging.getLogger(__name__)

# Zaratan SU charge per GPU-hour, matched on the GPU name (first match wins).
_SU_RATES = (("MIG", 7.0), ("H100", 144.0), ("A100", 48.0))

# Collective timeout. Rank 0 evaluates alone (greedy decoding of the eval sets,
# per-epoch pass@k) while the other ranks wait in a barrier; the default of
# 10 minutes would abort long evaluations.
DIST_TIMEOUT = datetime.timedelta(hours=2)


def lr_at(step: int, total: int, warmup: int, peak: float, min_ratio: float = 0.1) -> float:
    """
    Linear warmup, then cosine decay to `min_ratio * peak` at `total`.
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizer(model: torch.nn.Module, lr: float, wd: float) -> torch.optim.Optimizer:
    """
    AdamW with weight decay on matrices only (not norms / biases).
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=fused)


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def su_rate(override: float = 0.0) -> float:
    """
    SU per GPU-hour: `override` if set, else looked up from the GPU name.
    """
    if override > 0:
        return override
    name = gpu_name()
    for key, rate in _SU_RATES:
        if key in name:
            return rate
    return 0.0


def init_wandb(project: str, name: str, out_dir: str, config: dict, resume: bool = False):
    """
    Start a W&B run that mirrors the TensorBoard logs, or return None.

    Must be called before creating the `SummaryWriter`. Compute nodes cannot
    reach W&B, so runs default to offline mode; upload them later from a
    login node with `sync_wandb.sh`.
    """
    if not project:
        return None
    import wandb

    kwargs = {}
    if resume:
        # Keep one W&B run per output dir across resubmissions.
        id_path = os.path.join(out_dir, "wandb_run_id.txt")
        if os.path.exists(id_path):
            with open(id_path) as f:
                run_id = f.read().strip()
        else:
            run_id = uuid.uuid4().hex[:12]
            with open(id_path, "w") as f:
                f.write(run_id)
        kwargs = {"id": run_id, "resume": "allow"}
    return wandb.init(
        project=project,
        name=name or os.path.basename(os.path.normpath(out_dir)),
        dir=out_dir,
        config=config,
        mode=os.environ.get("WANDB_MODE", "offline"),
        sync_tensorboard=True,
        **kwargs,
    )


@dataclasses.dataclass
class Dist:
    """
    Process-group info; `world == 1` for plain single-process runs.
    """

    rank: int = 0
    world: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world > 1:
            dist.barrier()

    def mean(self, x: float) -> float:
        """
        Average a Python float over ranks.
        """
        if self.world == 1:
            return x
        t = torch.tensor([x], device=self.device)
        dist.all_reduce(t)
        return t.item() / self.world


def setup_dist() -> Dist:
    """
    Initialize DDP when launched with torchrun, else a single-process setup.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl", timeout=DIST_TIMEOUT)
            device = torch.device("cuda", local_rank)
        else:
            # CPU (gloo) so the multi-process path can be tested on CPU nodes.
            dist.init_process_group("gloo", timeout=DIST_TIMEOUT)
            device = torch.device("cpu")
        return Dist(dist.get_rank(), world, local_rank, device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Dist(device=device)


def cleanup_dist(d: Dist) -> None:
    if d.world > 1:
        dist.destroy_process_group()


def bf16_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return a bf16 copy of the model weights, leaving the model untouched.

    Tensors shared by several keys (tied embeddings) stay shared in the copy,
    so `save_pretrained` still recognizes them as tied.
    """
    copies: Dict[int, torch.Tensor] = {}
    out: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        ptr = v.data_ptr()
        if ptr not in copies:
            copies[ptr] = v.detach().to(torch.bfloat16)
        out[k] = copies[ptr]
    return out


def save_checkpoint(
    path: str, model: torch.nn.Module, opt: Optional[torch.optim.Optimizer], step: int,
    weights_only: bool, train_time_s: float = 0.0,
) -> None:
    """
    Save a resume checkpoint (atomically). Weights-only checkpoints store bf16
    weights without optimizer state (~2 bytes/param instead of ~16).

    :param train_time_s: training wall time up to `step`, summed over all jobs
        of the run, so the SU cost stays correct across resubmissions
    """
    state = {"step": step, "train_time_s": train_time_s}
    if weights_only:
        state["model"] = bf16_state_dict(model)
    else:
        state["model"] = model.state_dict()
        state["opt"] = opt.state_dict()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str, model: torch.nn.Module, opt: torch.optim.Optimizer, device
) -> Tuple[int, float]:
    """
    Load a checkpoint saved by `save_checkpoint`.

    :return: step and training wall time so far (0 for older checkpoints)
    """
    ck = torch.load(path, map_location=device)
    model.load_state_dict({k: v.to(torch.float32) for k, v in ck["model"].items()})
    if "opt" in ck:
        opt.load_state_dict(ck["opt"])
    else:
        _LOG.warning("Weights-only checkpoint: optimizer state restarts from zero")
    return int(ck["step"]), float(ck.get("train_time_s", 0.0))


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """
    Arguments shared by training scripts.
    """
    ap.add_argument("--su-per-gpu-hour", type=float, default=0.0,
                    help="SU rate for the cost estimate (0 = detect from the GPU name)")
    ap.add_argument("--wandb-project", default="",
                    help="also log to Weights & Biases (offline unless WANDB_MODE is set)")
    ap.add_argument("--wandb-name", default="", help="W&B run name (default: out dir name)")
    ap.add_argument("--no-save", action="store_true", help="save nothing (benchmarks)")
    ap.add_argument("--save-dtype", choices=("bf16", "fp32"), default="bf16",
                    help="dtype of the final saved weights")
    ap.add_argument("--ckpt-weights-only", action="store_true",
                    help="resume checkpoints without optimizer state (for big models)")
    ap.add_argument("--ledger", default="results/runs.jsonl",
                    help="run ledger (JSONL) that every run appends its metrics to")
