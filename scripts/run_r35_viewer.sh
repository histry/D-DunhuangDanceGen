#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?receipt}"
test $(cat "${root}_preview02_receipts/exit_status.txt") = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo targeted_tests > "$receipt/phase.txt"
"$native" -m pytest tests/test_transition_r35.py -q > "$receipt/tests.log" 2>&1
echo final_display_package > "$receipt/phase.txt"
"$native" -m dunhuang.research.transition_r35.bundle --root "$root" --out "$root/viewer02" > "$receipt/bundle.log" 2>&1
echo completed > "$receipt/phase.txt"
