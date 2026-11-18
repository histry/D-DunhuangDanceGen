#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/clock_residual_r55_02
exec > >(tee "$ROOT/receipts/comparison.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/comparison_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/comparison_receipt.txt"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/disk/lsm/conda_envs/edge/bin/python
set -x
for ARM in NoClock Unconditional Shuffled; do
 "$PY" -m dunhuang.research.clock_residual_r55 train --arm "$ARM" --out "$ROOT/$ARM"
done
"$PY" -m dunhuang.research.clock_residual_r55 evaluate --out "$ROOT/evaluation"
"$PY" -m dunhuang.research.clock_residual_r55 verify --out "$ROOT/independent_final"
CUDA_VISIBLE_DEVICES='' "$PY" -m pytest -q > "$ROOT/receipts/pytest_final.log" 2>&1
