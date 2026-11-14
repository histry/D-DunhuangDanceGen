#!/usr/bin/env bash
set -euo pipefail
study="$1"; receipt="${study}_generate_receipts"
edge=/home/disk/lsm/conda_envs/edge/bin/python
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "${study}_train_receipts/exit_status.txt")" = 0
test ! -e "$receipt"; mkdir "$receipt"
finish(){ code=$?; echo "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
cfg=src/dunhuang/research/motion_bridge/config.json
"$edge" -m dunhuang.research.motion_bridge.generate --config "$cfg" --training "$study" --out "$study/generation"
"$edge" -m dunhuang.research.motion_bridge.generate --config "$cfg" --training "$study" --out "$study/generation_replay" --replay-from "$study/generation"
"$native" -m dunhuang.research.motion_bridge.analyze --config "$cfg" --generation "$study/generation" --out "$study/analysis"
"$native" -m dunhuang.research.motion_bridge.analyze --config "$cfg" --generation "$study/generation_replay" --out "$study/analysis_replay" --replay-from "$study/analysis"
