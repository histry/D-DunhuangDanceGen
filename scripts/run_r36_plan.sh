#!/usr/bin/env bash
set -euo pipefail
root="${1:?fresh root}";receipt="${2:?fresh receipt}"
test ! -e "$root";test ! -e "$receipt";mkdir -p "$root" "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py=/home/disk/lsm/conda_envs/dance-rag/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo tests > "$receipt/phase.txt"
"$py" -m pytest tests/test_complete_r36.py -q > "$receipt/tests.log" 2>&1
echo planning > "$receipt/phase.txt"
"$py" -m dunhuang.research.complete_r36.study --out "$root/planning" > "$receipt/planning.log" 2>&1
echo replay > "$receipt/phase.txt"
"$py" -m dunhuang.research.complete_r36.study --out "$root/planning_replay" --replay-from "$root/planning" > "$receipt/replay.log" 2>&1
echo viewer > "$receipt/phase.txt"
"$py" -m dunhuang.research.complete_r36.view --root "$root" --out "$root/viewer_plan" > "$receipt/viewer.log" 2>&1
echo completed > "$receipt/phase.txt"
