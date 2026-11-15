#!/usr/bin/env bash
set -euo pipefail
study="${1:?study}";receipt="${2:?fresh receipts}"
training_receipt="${3:-${study}_training_receipts}"
test "$(cat "$training_receipt/exit_status.txt")" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cfg=src/dunhuang/research/reference_r26/config.json
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo compress > "$receipt/phase.txt"
"$edge" -m dunhuang.research.reference_r26.compress --config "$cfg" --root "$study" --out "$study/compress"
"$edge" -m dunhuang.research.reference_r26.compress --config "$cfg" --root "$study" --out "$study/compress_replay" --replay-from "$study/compress"
echo samples > "$receipt/phase.txt"
"$edge" -m dunhuang.research.reference_r26.sample --config "$cfg" --root "$study" --out "$study/samples"
"$edge" -m dunhuang.research.reference_r26.sample --config "$cfg" --root "$study" --out "$study/samples_replay" --replay-from "$study/samples"
echo metrics > "$receipt/phase.txt"
"$native" -m dunhuang.research.reference_r26.evaluate --config "$cfg" --root "$study" --out "$study/evaluation"
"$native" -m dunhuang.research.reference_r26.evaluate --config "$cfg" --root "$study" --out "$study/evaluation_replay" --replay-from "$study/evaluation"
echo completed > "$receipt/phase.txt"
