#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/music_structure_r56_02
exec > >(tee "$ROOT/receipts/controls.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/controls_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/controls_receipt.txt"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/disk/lsm/conda_envs/edge/bin/python
set -x
"$PY" -c "import json; assert json.load(open('$ROOT/independent_main/result.json'))['exact_arrays_bytes']; assert json.load(open('$ROOT/receipts/main_browser.json'))['passed']"
"$PY" -m dunhuang.research.music_structure_r56 controls --out "$ROOT/controls"
"$PY" -m dunhuang.research.music_structure_r56 verify --source "$ROOT/controls" --out "$ROOT/independent_controls"
"$PY" -m dunhuang.research.music_structure_r56 blind --source "$ROOT/main" --out "$ROOT/blind"
"$PY" -m pytest -q > "$ROOT/receipts/pytest_final.log" 2>&1
