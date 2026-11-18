#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/tempo_attribution_r57_01
exec > >(tee "$ROOT/receipts/repair.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/repair_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/repair_receipt.txt"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/disk/lsm/conda_envs/edge/bin/python
set -x
"$PY" -m dunhuang.research.geometry_recovery_r57.study run --out "$ROOT/repair"
"$PY" -m dunhuang.research.geometry_recovery_r57.study verify --source "$ROOT/repair" --out "$ROOT/independent_repair"
"$PY" -m pytest -q > "$ROOT/receipts/pytest_final_repair.log" 2>&1
