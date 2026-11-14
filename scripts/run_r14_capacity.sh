#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_capacity_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "${study}_readout_receipts/exit_status.txt")" = 0
test ! -e "$receipt";mkdir "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
cfg=src/dunhuang/research/boundary_state_r14/capacity_config.json
"$edge" -m dunhuang.research.boundary_state_r14.capacity --config "$cfg" --root "$study" --out "$study/capacity"
"$edge" -m dunhuang.research.boundary_state_r14.capacity --config "$cfg" --root "$study" --out "$study/capacity_replay" --replay-from "$study/capacity"
"$native" -m dunhuang.research.boundary_state_r14.capacity_native --config "$cfg" --root "$study" --out "$study/capacity_native"
"$native" -m dunhuang.research.boundary_state_r14.capacity_native --config "$cfg" --root "$study" --out "$study/capacity_native_replay" --replay-from "$study/capacity_native"
