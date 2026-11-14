#!/usr/bin/env bash
set -euo pipefail
study="$1"; receipt="${study}_chain_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "${study}_final_receipts/exit_status.txt")" = 0
test ! -e "$receipt"; mkdir "$receipt"
finish(){ code=$?; echo "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
"$native" -m dunhuang.research.motion_bridge.verify_chain --generation "$study/generation" --out "$study/chain"
"$native" -m dunhuang.research.motion_bridge.verify_chain --generation "$study/generation_replay" --out "$study/chain_replay" --replay-from "$study/chain"
cp "$study/independent_checks.json" "$study/pre_chain_independent_checks.json"
"$native" scripts/verify_r12.py --root "$study"
