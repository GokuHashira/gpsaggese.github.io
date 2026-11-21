"""
Run metrics: GPU monitoring, compute accounting and a run ledger.

Every training / evaluation script records the same facts about itself so
that runs can be compared on cost as well as quality:

- `GpuMonitor` samples utilization, memory and power of every visible GPU in a
  background thread (via `nvidia-smi`) and integrates energy (kWh).
- `flops_per_token`, `mfu`: model FLOPs per token and model FLOPs utilization.
- `slurm_info`: job id, partition, node, submit / start time, queue wait.
- `RunLedger`: appends one JSON record per run to `results/runs.jsonl`
  (file-locked, safe for concurrent jobs).

Import as:

import logic_slm_metrics as lsmet
"""

import datetime
import fcntl
import json
import logging
import os
import socket
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

_LOG = logging.getLogger(__name__)

# Dense bf16 peak FLOP/s per GPU (no sparsity), matched on the GPU name.
_PEAK_BF16 = (("MIG 1g", 312e12 / 7), ("H100", 989e12), ("A100", 312e12), ("V100", 125e12))
DEFAULT_LEDGER = "results/runs.jsonl"


def peak_flops(gpu_name: str) -> float:
    """
    Dense bf16 peak FLOP/s for one GPU (0 if unknown, e.g. CPU).
    """
    for key, val in _PEAK_BF16:
        if key in gpu_name:
            return val
    return 0.0


def flops_per_token(non_embedding_params: int, n_layers: int, hidden: int, seq_len: int,
                    vocab: int = 0) -> float:
    """
    Training FLOPs per token: 6 * N (weights, forward + backward) plus attention
    score/value FLOPs 12 * L * d * T (Kaplan et al. / PaLM accounting), plus the
    output projection 6 * d * V when the vocabulary is given.
    """
    return 6 * non_embedding_params + 12 * n_layers * hidden * seq_len + 6 * hidden * vocab


def mfu(tokens_per_s: float, fpt: float, n_gpus: int, gpu_name: str) -> float:
    """
    Model FLOPs utilization: achieved model FLOP/s over the GPUs' peak.
    """
    peak = peak_flops(gpu_name) * max(n_gpus, 1)
    return tokens_per_s * fpt / peak if peak else 0.0


def git_commit(path: str = ".") -> str:
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:  # pragma: no cover - git missing
        return "unknown"


def _parse_slurm_time(s: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError):
        return None


def slurm_info() -> Dict[str, Any]:
    """
    Slurm facts about the current job (empty outside Slurm).
    """
    job = os.environ.get("SLURM_JOB_ID")
    info: Dict[str, Any] = {"host": socket.gethostname()}
    if not job:
        return info
    info.update({
        "slurm_job_id": job,
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
    })
    try:
        out = subprocess.run(["scontrol", "show", "job", job], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, timeout=20).stdout
        fields = dict(tok.split("=", 1) for tok in out.split() if "=" in tok)
        submit, start = _parse_slurm_time(fields.get("SubmitTime")), _parse_slurm_time(fields.get("StartTime"))
        info["slurm_submit_time"] = fields.get("SubmitTime")
        info["slurm_start_time"] = fields.get("StartTime")
        info["slurm_time_limit"] = fields.get("TimeLimit")
        if submit and start:
            info["queue_wait_s"] = (start - submit).total_seconds()
    except Exception:  # pragma: no cover - scontrol missing
        pass
    return info


class GpuMonitor:
    """
    Background sampler of per-GPU utilization, memory and power.

    Writes `<out_dir>/gpu_trace.csv` and integrates energy. Does nothing (and
    reports zeros) when `nvidia-smi` is unavailable.
    """

    QUERY = "index,utilization.gpu,memory.used,power.draw"

    def __init__(self, out_dir: str, interval_s: float = 15.0) -> None:
        self.path = os.path.join(out_dir, "gpu_trace.csv")
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[Dict[str, float]] = []
        self.energy_j = 0.0
        self._last_t: Optional[float] = None

    def _sample(self) -> List[Dict[str, float]]:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=20,
        ).stdout
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                idx, util, mem, power = (float(p) if p not in ("[N/A]", "N/A") else 0.0 for p in parts)
            except ValueError:
                continue
            rows.append({"gpu": idx, "util": util, "mem_mib": mem, "power_w": power})
        return rows

    def _run(self) -> None:
        with open(self.path, "w") as f:
            f.write("time,gpu,util_pct,mem_mib,power_w\n")
            while not self._stop.is_set():
                t = time.time()
                try:
                    rows = self._sample()
                except Exception:
                    rows = []
                if self._last_t is not None and rows:
                    self.energy_j += sum(r["power_w"] for r in rows) * (t - self._last_t)
                self._last_t = t
                for r in rows:
                    f.write(f"{t:.0f},{r['gpu']:.0f},{r['util']:.0f},{r['mem_mib']:.0f},{r['power_w']:.1f}\n")
                    self.samples.append(r)
                f.flush()
                self._stop.wait(self.interval)

    def start(self) -> "GpuMonitor":
        try:
            subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=20, check=True)
        except Exception:
            _LOG.info("nvidia-smi not available: GPU monitoring disabled")
            return self
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, float]:
        """
        Stop sampling and return summary statistics.
        """
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=30)
        if not self.samples:
            return {"gpu_util_mean_pct": 0.0, "gpu_mem_peak_mib": 0.0, "gpu_power_mean_w": 0.0, "energy_kwh": 0.0}
        n = len(self.samples)
        return {
            "gpu_util_mean_pct": sum(s["util"] for s in self.samples) / n,
            "gpu_mem_peak_mib": max(s["mem_mib"] for s in self.samples),
            "gpu_power_mean_w": sum(s["power_w"] for s in self.samples) / n,
            "energy_kwh": self.energy_j / 3.6e6,
        }


class Timer:
    """
    Named wall-clock phases: `with timer.phase("compile"): ...`.
    """

    def __init__(self) -> None:
        self.t0 = time.time()
        self.phases: Dict[str, float] = {}

    class _Phase:
        def __init__(self, timer: "Timer", name: str) -> None:
            self.timer, self.name = timer, name

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *exc):
            self.timer.add(self.name, time.time() - self.start)
            return False

    def phase(self, name: str) -> "_Phase":
        return Timer._Phase(self, name)

    def add(self, name: str, seconds: float) -> None:
        self.phases[name] = self.phases.get(name, 0.0) + seconds

    def total(self) -> float:
        return time.time() - self.t0


def cost_fields(wall_s: float, phases: Dict[str, float], n_gpus: int, su_per_gpu_hour: float) -> Dict[str, float]:
    """
    GPU-hours and SU, overall and per phase.
    """
    out = {"wall_time_s": wall_s, "n_gpus": n_gpus, "gpu_hours": wall_s / 3600 * n_gpus,
           "su_per_gpu_hour": su_per_gpu_hour, "su": wall_s / 3600 * n_gpus * su_per_gpu_hour}
    for name, sec in phases.items():
        out[f"time_{name}_s"] = sec
        out[f"su_{name}"] = sec / 3600 * n_gpus * su_per_gpu_hour
    return out


class RunLedger:
    """
    Append-only JSONL ledger of runs (one record per run), safe for concurrent writers.
    """

    def __init__(self, path: str = DEFAULT_LEDGER) -> None:
        self.path = path

    def append(self, record: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        rec = {"recorded_at": datetime.datetime.now().isoformat(timespec="seconds"), **record}
        line = json.dumps(rec, default=str) + "\n"
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
            fcntl.flock(f, fcntl.LOCK_UN)

    def read(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]
