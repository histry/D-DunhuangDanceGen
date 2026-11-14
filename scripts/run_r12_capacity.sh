#!/usr/bin/env bash
set -euo pipefail
study="$1"; receipt="${study}_capacity_receipts"
edge=/home/disk/lsm/conda_envs/edge/bin/python
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test ! -e "$study"; test ! -e "$receipt"; mkdir -p "$study" "$receipt"
finish(){ code=$?; echo "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_motion_bridge.py > "$receipt/edge_tests.log" 2>&1
cfg=src/dunhuang/research/motion_bridge/config.json
"$edge" -m dunhuang.research.motion_bridge.study capacity --config "$cfg" --out "$study/capacity"
"$edge" -m dunhuang.research.motion_bridge.study capacity --config "$cfg" --out "$study/capacity_replay" --replay-from "$study/capacity"
