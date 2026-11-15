#!/usr/bin/env bash
set -euo pipefail
study="${1:?fresh root}";receipt="${2:?fresh receipts}"
test ! -e "$study";test ! -e "$receipt";mkdir -p "$study" "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?; echo "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; exit "$code"' EXIT
cfg=src/dunhuang/research/lodge_reference_r24/config.json
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_lodge_reference_r24.py tests/test_local_r23.py > "$receipt/edge_tests.log" 2>&1
echo frozen_lodge > "$receipt/phase.txt"
"$edge" -m dunhuang.research.lodge_reference_r24.experiment --config "$cfg" --out "$study/main"
echo independent_replay > "$receipt/phase.txt"
"$edge" -m dunhuang.research.lodge_reference_r24.experiment --config "$cfg" --out "$study/replay" --replay-from "$study/main"
echo completed > "$receipt/phase.txt"
