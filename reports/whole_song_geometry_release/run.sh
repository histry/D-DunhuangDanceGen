#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/whole_song_geometry_02
mkdir -p "$ROOT/receipts"
exec > >(tee "$ROOT/receipts/run.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/run_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/run_receipt.txt"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/disk/lsm/conda_envs/edge/bin/python
set -x
"$PY" -m pytest -q tests/test_whole_song_geometry.py
"$PY" -m dunhuang.research.whole_song_geometry run --out "$ROOT/main"
"$PY" -m dunhuang.research.whole_song_geometry verify --out "$ROOT/independent" --source "$ROOT/main"
