#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test "$(cat ${root}_preview_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/coverage_r34/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo tests > "$receipt/phase.txt"
"$edge" -m pytest tests/test_coverage_r34.py tests/test_plan_view_r34.py -q > "$receipt/edge_tests.log" 2>&1
echo fresh_E1W_and_frozen_score_binding > "$receipt/phase.txt"
"$native" -m dunhuang.research.plan_view_r34.planning --root "$root" --out "$root/planning"
"$native" -m dunhuang.research.plan_view_r34.planning --root "$root" --out "$root/planning_replay" --replay-from "$root/planning"
echo actual_plan_bound_windows > "$receipt/phase.txt"
"$edge" -m dunhuang.research.plan_view_r34.generate --config "$cfg" --root "$root" --out "$root/plan_generation"
echo plan_generation_fresh_replay > "$receipt/phase.txt"
"$edge" -m dunhuang.research.plan_view_r34.generate --config "$cfg" --root "$root" --out "$root/plan_generation_replay" --replay-from "$root/plan_generation"
echo completed > "$receipt/phase.txt"
