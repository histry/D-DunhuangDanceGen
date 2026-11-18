#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/tempo_attribution_r57_01
exec > >(tee "$ROOT/receipts/main.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/main_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/main_receipt.txt"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/disk/lsm/conda_envs/edge/bin/python
set -x
"$PY" -m pytest -q tests/test_tempo_attribution_r57.py tests/test_music_structure_r56.py
"$PY" -c "from dunhuang.research.tempo_attribution_r57.common import verify_parent; print('R56_PARENT_SEALED',verify_parent())"
"$PY" -m dunhuang.research.tempo_attribution_r57 main --out "$ROOT/main"
"$PY" -m dunhuang.research.tempo_attribution_r57 render --source "$ROOT/main" --out "$ROOT/main_video"
"$PY" -m dunhuang.research.tempo_attribution_r57 verify --source "$ROOT/main" --out "$ROOT/independent_main"
