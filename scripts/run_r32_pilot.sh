#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test "$(cat ${root}_initial_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/content_r32/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo partial_contract_tests > "$receipt/phase.txt"
"$edge" -m pytest tests/test_content_r32.py -q > "$receipt/edge_tests.log" 2>&1
echo pilot > "$receipt/phase.txt"
"$edge" -m dunhuang.research.content_r32.train --config "$cfg" --root "$root" --out "$root/models/pilot" --arm pilot
"$edge" -m dunhuang.research.content_r32.train --config "$cfg" --root "$root" --out "$root/models/pilot_replay" --arm pilot --replay-from "$root/models/pilot"
echo completed > "$receipt/phase.txt"
