#!/usr/bin/env bash
set -euo pipefail
main="$1"
b2="$2"
receipt="$main"_post_receipts
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "$main"_receipts/exit_status.txt)" = 0
test "$(cat "$b2"_receipts/exit_status.txt)" = 0
test ! -e "$receipt"; mkdir "$receipt"
finish() { code=$?; echo "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
date -u +%FT%TZ > "$receipt/started_utc.txt"
git rev-parse HEAD > "$receipt/execution_commit.txt"
"$native" -m dunhuang.research.feasible_reperformance.analyse --main "$main" --out "$main/analysis"
"$native" -m dunhuang.research.feasible_reperformance.analyse --main "$main" --out "$main/analysis_replay" --no-render
"$native" scripts/verify_r9_media.py --main "$main" > "$receipt/media_checks.log" 2>&1
