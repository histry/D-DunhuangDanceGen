#!/usr/bin/env bash
set -euo pipefail
study="$1"
receipt="${study}_post_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH="$PWD/src"
test "$(cat "${study}_receipts/exit_status.txt")" = 0
test ! -e "$receipt"
mkdir "$receipt"
success=0
finish() { code=$?; if [ "$success" -ne 1 ] && [ "$code" -eq 0 ]; then code=125; fi; printf '%s\n' "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
date -u +%FT%TZ > "$receipt/started_utc.txt"
git rev-parse HEAD > "$receipt/execution_commit.txt"
cp "$study/public_checks.json" "$receipt/previous_public_checks.json"
"$native" -m dunhuang.research.song_transfer.diagnosis --generation "$study/generation" --out "$study/diagnosis"
"$native" -m dunhuang.research.song_transfer.diagnosis --generation "$study/generation" --out "$study/diagnosis_replay" --replay-from "$study/diagnosis"
"$native" -m dunhuang.research.song_transfer.render --generation "$study/generation" --features "$study/features" --out "$study/videos_native_plane"
"$native" scripts/check_full_song_transfer.py --out "$study" > "$receipt/independent_checks.log" 2>&1
success=1
