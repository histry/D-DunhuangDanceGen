#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test "$(cat ${root}_main_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo tests > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest tests/test_content_r32.py -q > "$receipt/edge_tests.log" 2>&1
echo matched_noise_music_probe > "$receipt/phase.txt"
"$edge" -m dunhuang.research.content_r32.music_probe --config src/dunhuang/research/content_r32/config.json --root "$root" --out "$root/music_probe"
"$edge" -m dunhuang.research.content_r32.music_probe --config src/dunhuang/research/content_r32/config.json --root "$root" --out "$root/music_probe_replay" --replay-from "$root/music_probe"
echo independent_checks > "$receipt/phase.txt"
"$native" scripts/release_r32.py --task verify --root "$root"
echo render > "$receipt/phase.txt"
"$edge" scripts/release_r32.py --task render --root "$root" --out "$root/videos"
echo completed > "$receipt/phase.txt"
