#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test "$(cat ${root}_preflight_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/control_r33/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo tests > "$receipt/phase.txt"
"$edge" -m pytest tests/test_control_r33.py -q > "$receipt/edge_tests.log" 2>&1
echo train_only_scales_gradients_finite_updates > "$receipt/phase.txt"
"$edge" -m dunhuang.research.control_r33.calibrate --config "$cfg" --root "$root" --out "$root/calibration"
"$edge" -m dunhuang.research.control_r33.calibrate --config "$cfg" --root "$root" --out "$root/calibration_replay" --replay-from "$root/calibration"
echo completed > "$receipt/phase.txt"
