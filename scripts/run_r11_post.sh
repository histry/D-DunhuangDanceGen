#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="$study"_post_receipts
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "$study"_trace_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
cfg=src/dunhuang/research/state_bridge/config.json
"$native" -m dunhuang.research.state_bridge.analyze --config "$cfg" --audit "$study/audit" --traces "$study/trace" --out "$study/analysis"
"$native" -m dunhuang.research.state_bridge.analyze --config "$cfg" --audit "$study/audit_replay" --traces "$study/trace_replay" --out "$study/analysis_replay" --replay-from "$study/analysis"
"$native" scripts/verify_r11.py --root "$study"
"$native" -m dunhuang.research.state_bridge.media --config "$cfg" --audit "$study/audit" --analysis "$study/analysis" --out "$study/inspector"
