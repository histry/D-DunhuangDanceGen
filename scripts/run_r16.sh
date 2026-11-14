#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_receipts"
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
"$edge" -m pytest -q tests/test_continuous_r16.py > "$receipt/edge_tests.log" 2>&1
cfg=src/dunhuang/research/continuous_r16/config.json
"$native" -m dunhuang.research.continuous_r16.preflight --config "$cfg" --out "$study/preflight"
"$native" -m dunhuang.research.continuous_r16.preflight --config "$cfg" --out "$study/preflight_replay" --replay-from "$study/preflight"
"$edge" -m dunhuang.research.continuous_r16.sample --config "$cfg" --preflight "$study/preflight" --out "$study/samples"
"$edge" -m dunhuang.research.continuous_r16.sample --config "$cfg" --preflight "$study/preflight" --out "$study/samples_replay" --replay-from "$study/samples"
"$native" -m dunhuang.research.continuous_r16.evaluate --config "$cfg" --preflight "$study/preflight" --samples "$study/samples" --out "$study/native"
"$native" -m dunhuang.research.continuous_r16.evaluate --config "$cfg" --preflight "$study/preflight" --samples "$study/samples" --out "$study/native_replay" --replay-from "$study/native"
