#!/usr/bin/env bash
set -euo pipefail
study="$1"
receipt="$study"_receipts
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test ! -e "$study"; test ! -e "$receipt"; mkdir -p "$study" "$receipt"
finish() { status=$?; echo "$status" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%FT%TZ > "$receipt/started_utc.txt"
git rev-parse HEAD > "$receipt/execution_commit.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q -rs tests/test_primitive_reperformance.py tests/test_bridge_inpainting.py tests/test_boundary_state.py tests/test_feasible_reperformance.py > "$receipt/edge_tests.log" 2>&1
"$native" -m dunhuang.research.boundary_state.preflight --out "$study/training_preflight"
b=src/dunhuang/research/boundary_state/config.json
s=src/dunhuang/research/feasible_reperformance/config.json
"$native" -m dunhuang.research.boundary_state.study --config "$b" --out "$study/boundary"
"$native" -m dunhuang.research.boundary_state.study --config "$b" --out "$study/boundary_replay" --replay-from "$study/boundary"
"$native" -m dunhuang.research.feasible_reperformance.study --config "$s" --out "$study/source"
"$native" -m dunhuang.research.feasible_reperformance.study --config "$s" --out "$study/source_replay" --replay-from "$study/source"
"$native" scripts/verify_r9.py --out "$study" > "$receipt/independent_checks.log" 2>&1
