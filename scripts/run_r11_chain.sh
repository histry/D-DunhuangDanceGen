#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="$study"_chain_receipts
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "$study"_final_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
cfg=src/dunhuang/research/state_bridge/config.json
"$native" -m dunhuang.research.state_bridge.stage_chain --config "$cfg" --trace "$study/trace" --out "$study/chain"
"$native" -m dunhuang.research.state_bridge.stage_chain --config "$cfg" --trace "$study/trace_replay" --out "$study/chain_replay" --replay-from "$study/chain"
"$native" scripts/verify_r11.py --root "$study" > "$receipt/independent_checks.log" 2>&1
