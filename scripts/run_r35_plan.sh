#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?receipt}"
test ! -e "$root";test ! -e "$receipt";mkdir -p "$root" "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py=/home/disk/lsm/conda_envs/dance-rag/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo tests > "$receipt/phase.txt"
"$py" -m pytest tests/test_transition_r35.py -q > "$receipt/pytest.log" 2>&1
echo planning > "$receipt/phase.txt"
"$py" -m dunhuang.research.transition_r35.planning --out "$root/planning" > "$receipt/planning.log" 2>&1
echo independent_planning_replay > "$receipt/phase.txt"
"$py" -m dunhuang.research.transition_r35.planning --out "$root/planning_replay" --replay-from "$root/planning" > "$receipt/replay.log" 2>&1
echo completed > "$receipt/phase.txt"
