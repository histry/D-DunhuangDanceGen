#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?receipt}"
test -f "$root/planning/result.json";test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo tests > "$receipt/phase.txt"
"$native" -m pytest tests/test_transition_r35.py -q > "$receipt/tests.log" 2>&1
"$edge" -m pytest tests/test_coverage_r34.py tests/test_plan_view_r34.py tests/test_transition_r35.py -q > "$receipt/edge_tests.log" 2>&1
echo train > "$receipt/phase.txt"
"$edge" -m dunhuang.research.transition_r35.train --root "$root" --out "$root/context" > "$receipt/train.log" 2>&1
echo independent_replay > "$receipt/phase.txt"
"$edge" -m dunhuang.research.transition_r35.train --root "$root" --out "$root/context_replay" --replay-from "$root/context" > "$receipt/replay.log" 2>&1
echo completed > "$receipt/phase.txt"
