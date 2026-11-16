#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipts}"
test "$(cat ${root}_evaluation_receipts/exit_status.txt)" = 0
test "$(cat ${root}_coverage_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo independent_checks > "$receipt/phase.txt"
"$native" scripts/release_r29.py --task verify --root "$root"
echo render > "$receipt/phase.txt"
"$native" scripts/release_r29.py --task render --root "$root" --out "$root/videos"
echo completed > "$receipt/phase.txt"