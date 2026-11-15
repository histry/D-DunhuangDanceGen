#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipts}"
test "$(cat ${root}_training_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/contract_r28/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo population_gate > "$receipt/phase.txt"
"$native" -m dunhuang.research.contract_r28.decision --root "$root"
echo ChangE > "$receipt/phase.txt"
"$edge" -m dunhuang.research.contract_r28.change --config "$cfg" --root "$root" --out "$root/change"
"$edge" -m dunhuang.research.contract_r28.change --config "$cfg" --root "$root" --out "$root/change_replay" --replay-from "$root/change"
echo independent_checks > "$receipt/phase.txt"
"$native" scripts/release_r28.py --task verify --root "$root"
echo render > "$receipt/phase.txt"
"$native" scripts/release_r28.py --task render --root "$root" --out "$root/videos"
echo completed > "$receipt/phase.txt"
