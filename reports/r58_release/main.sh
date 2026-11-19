#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/execution_planning_r58_01
exec > >(tee "$ROOT/receipts/main.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/main_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/main_receipt.txt"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/disk/lsm/conda_envs/edge/bin/python
set -x
"$PY" -m pytest -q tests/test_execution_planning_r58.py tests/test_tempo_attribution_r57.py tests/test_geometry_recovery_r57.py
"$PY" -c "import runpy;from pathlib import Path;f=runpy.run_path('scripts/release_r55_and_song.py')['validate_seal'];print('PARENT_SEALS',f(Path('outputs/music_structure_r56_02')),f(Path('outputs/tempo_attribution_r57_01')))"
"$PY" -m dunhuang.research.execution_planning_r58 main --out "$ROOT/main"
"$PY" -m dunhuang.research.execution_planning_r58 render --source "$ROOT/main" --out "$ROOT/main_video"
"$PY" -m dunhuang.research.execution_planning_r58 verify --source "$ROOT/main" --out "$ROOT/independent_main"
