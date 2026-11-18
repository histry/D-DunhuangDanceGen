#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/tempo_attribution_r57_01
exec > >(tee "$ROOT/receipts/controls.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/controls_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/controls_receipt.txt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/disk/lsm/conda_envs/edge/bin/python
set -x
"$PY" -c "import json; assert json.load(open('$ROOT/receipts/main_browser.json'))['passed']; assert json.load(open('$ROOT/independent_main/result.json'))['arrays_bytes_exact']"
CUDA_VISIBLE_DEVICES='' "$PY" -m dunhuang.research.tempo_attribution_r57 controls --out "$ROOT/controls"
CUDA_VISIBLE_DEVICES='' "$PY" -m dunhuang.research.tempo_attribution_r57 verify --source "$ROOT/controls" --out "$ROOT/independent_controls"
CUDA_VISIBLE_DEVICES='' "$PY" -c "import runpy; print('R55_SEAL',runpy.run_path('scripts/release_r55_and_song.py')['validate_seal'](__import__('pathlib').Path('outputs/clock_residual_r55_02')))"
CUDA_VISIBLE_DEVICES=0 "$PY" -m dunhuang.research.tempo_attribution_r57 sdmr --out "$ROOT/sdmr"
CUDA_VISIBLE_DEVICES=0 "$PY" -m dunhuang.research.tempo_attribution_r57 verify_sdmr --source "$ROOT/sdmr" --out "$ROOT/independent_sdmr"
CUDA_VISIBLE_DEVICES='' "$PY" -m pytest -q > "$ROOT/receipts/pytest_final.log" 2>&1
