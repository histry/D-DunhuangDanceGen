#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/music_structure_r56_02
mkdir -p "$ROOT/receipts"
exec > >(tee "$ROOT/receipts/main.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/main_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/main_receipt.txt"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/disk/lsm/conda_envs/edge/bin/python
set -x
"$PY" -m pytest -q tests/test_music_structure_r56.py tests/test_whole_song_geometry.py
"$PY" -m dunhuang.research.music_structure_r56 main --out "$ROOT/main"
"$PY" -m dunhuang.research.music_structure_r56 render --source "$ROOT/main" --out "$ROOT/main_video"
"$PY" -m dunhuang.research.music_structure_r56 verify --source "$ROOT/main" --out "$ROOT/independent_main"
