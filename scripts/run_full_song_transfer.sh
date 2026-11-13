#!/usr/bin/env bash
set -euo pipefail
study="$1"
receipt="$2"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH="$PWD/src"
test ! -e "$study"
test ! -e "$receipt"
mkdir -p "$study" "$receipt"
success=0
finish() { code=$?; if [ "$success" -ne 1 ] && [ "$code" -eq 0 ]; then code=125; fi; printf '%s\n' "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%FT%TZ > "$receipt/started_utc.txt"
git rev-parse HEAD > "$receipt/execution_commit.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
cfg=configs/exposed_full_song_r7_v1.json
"$native" -m dunhuang.research.song_transfer.music --config "$cfg" --out "$study/music"
"$native" -m dunhuang.research.song_transfer.music --config "$cfg" --out "$study/music_replay" --replay-from "$study/music"
"$edge" -m dunhuang.research.song_transfer.features --config "$cfg" --music "$study/music" --out "$study/features"
"$edge" -m dunhuang.research.song_transfer.features --config "$cfg" --music "$study/music" --out "$study/features" --verify
"$native" -m dunhuang.research.song_transfer.generation --config "$cfg" --music "$study/music" --features "$study/features" --out "$study/generation"
"$native" -m dunhuang.research.song_transfer.generation --config "$cfg" --music "$study/music" --features "$study/features" --out "$study/generation_replay" --replay-from "$study/generation"
"$native" -m dunhuang.research.song_transfer.render --generation "$study/generation" --features "$study/features" --out "$study/videos"
success=1
