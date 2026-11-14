#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="$study"_audit_receipts
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test ! -e "$study";test ! -e "$receipt";mkdir -p "$study" "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
cfg=src/dunhuang/research/state_bridge/config.json
"$native" -m dunhuang.research.state_bridge.audit --config "$cfg" --out "$study/audit"
"$native" -m dunhuang.research.state_bridge.audit --config "$cfg" --out "$study/audit_replay" --replay-from "$study/audit"
