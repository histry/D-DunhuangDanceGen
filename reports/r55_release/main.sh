#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/clock_residual_r55_02
mkdir -p "$ROOT/receipts"
exec > >(tee "$ROOT/receipts/main.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/main_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/main_receipt.txt"
git status --short
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/disk/lsm/conda_envs/edge/bin/python
set -x
"$PY" -m pytest -q tests/test_clock_residual_r55.py tests/test_r54.py
"$PY" -m dunhuang.research.clock_residual_r55 prepare --out "$ROOT/input"
"$PY" -m dunhuang.research.clock_residual_r55 preview --out "$ROOT/preview"
"$PY" -m dunhuang.research.clock_residual_r55 qualify --out "$ROOT/qualification"
if "$PY" -c "import json; assert json.load(open('$ROOT/qualification/gate.json'))['admitted']"; then
 "$PY" -m dunhuang.research.clock_residual_r55 train --arm Conditional --out "$ROOT/Conditional"
 "$PY" -m dunhuang.research.clock_residual_r55 main_view --out "$ROOT/main_view"
fi
"$PY" -m dunhuang.research.clock_residual_r55 verify --out "$ROOT/independent_main"
