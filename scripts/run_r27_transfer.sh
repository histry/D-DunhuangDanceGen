#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipts}"
test "$(cat ${root}_followup_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cfg=src/dunhuang/research/state_r27/config.json
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo admission > "$receipt/phase.txt"
"$native" -m dunhuang.research.state_r27.admit --root "$root"
echo transfer > "$receipt/phase.txt"
"$edge" -m dunhuang.research.state_r27.transfer --config "$cfg" --root "$root" --out "$root/transfer"
"$edge" -m dunhuang.research.state_r27.transfer --config "$cfg" --root "$root" --out "$root/transfer_replay" --replay-from "$root/transfer"
echo independent_checks > "$receipt/phase.txt"
"$native" scripts/release_r27.py --task verify --root "$root"
echo render > "$receipt/phase.txt"
"$native" scripts/release_r27.py --task render --root "$root" --out "$root/videos"
echo completed > "$receipt/phase.txt"
