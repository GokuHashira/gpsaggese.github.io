#!/bin/bash
# Upload offline W&B runs that have not been synced yet.
#
# Compute nodes cannot reach W&B, so `pretrain.py --wandb-project ...` logs
# offline into `<run dir>/wandb/offline-run-*`. Run this on a LOGIN node:
#
#   > ./sync_wandb.sh            # all runs under models/
#   > ./sync_wandb.sh models/e2  # only runs under models/e2
#
# Login nodes cap processes per user, so thread counts are limited to 1.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GOMAXPROCS=1
export PYTHONPATH=$HOME/msml610/pylibs${PYTHONPATH:+:$PYTHONPATH}
PY=/scratch/zt1/project/msml610/user/gsakthiv/envs/slm/bin/python
ROOT=${1:-models}
n=0
while IFS= read -r run; do
    # `wandb sync` leaves a marker file next to runs it has uploaded.
    if ls "$run"/*.wandb.synced >/dev/null 2>&1; then
        continue
    fi
    echo "Syncing $run"
    $PY -m wandb sync "$run"
    n=$((n + 1))
done < <(find "$ROOT" -type d -name "offline-run-*" | sort)
echo "Synced $n run(s)."
