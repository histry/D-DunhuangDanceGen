#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_audit_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test ! -e "$study";test ! -e "$receipt";mkdir -p "$study" "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_boundary_state_r14.py > "$receipt/edge_tests.log" 2>&1
cfg=src/dunhuang/research/boundary_state_r14/config.json
"$native" -m dunhuang.research.boundary_state_r14.audit --config "$cfg" --out "$study/audit"
"$native" -m dunhuang.research.boundary_state_r14.audit --config "$cfg" --out "$study/audit_replay" --replay-from "$study/audit"
"$edge" -m dunhuang.research.boundary_state_r14.conflict --config "$cfg" --audit "$study/audit" --out "$study/conflict"
"$edge" -m dunhuang.research.boundary_state_r14.conflict --config "$cfg" --audit "$study/audit" --out "$study/conflict_replay" --replay-from "$study/conflict"
