#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_readout_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "${study}_audit_receipts/exit_status.txt")" = 0
test ! -e "$receipt";mkdir "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
cfg=src/dunhuang/research/boundary_state_r14/config.json
"$native" -m dunhuang.research.boundary_state_r14.readout --config "$cfg" --out "$study/readout"
"$native" -m dunhuang.research.boundary_state_r14.readout --config "$cfg" --out "$study/readout_replay" --replay-from "$study/readout"
