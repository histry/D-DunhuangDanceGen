#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test "$(cat ${root}_train02_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/coverage_r34/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo tests > "$receipt/phase.txt"
"$edge" -m pytest tests/test_coverage_r34.py -q > "$receipt/edge_tests.log" 2>&1
echo primary_effect_full_sampling > "$receipt/phase.txt"
"$edge" -m dunhuang.research.coverage_r34.evaluate --config "$cfg" --root "$root" --out "$root/preview" --preview-only
echo primary_effect_fresh_replay > "$receipt/phase.txt"
"$edge" -m dunhuang.research.coverage_r34.evaluate --config "$cfg" --root "$root" --out "$root/preview_replay" --preview-only --replay-from "$root/preview"
echo completed > "$receipt/phase.txt"
