#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipts}"
test "$(cat ${root}_evaluation_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/time_r29/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo cohort > "$receipt/phase.txt"
"$edge" -m dunhuang.research.time_r29.coverage --task data --config "$cfg" --root "$root" --out "$root/coverage_data"
"$edge" -m dunhuang.research.time_r29.coverage --task data --config "$cfg" --root "$root" --out "$root/coverage_data_replay" --replay-from "$root/coverage_data"
echo training > "$receipt/phase.txt"
"$edge" -m dunhuang.research.time_r29.coverage --task train --config "$cfg" --root "$root" --out "$root/models/COVER"
"$edge" -m dunhuang.research.time_r29.coverage --task train --config "$cfg" --root "$root" --out "$root/models/COVER_replay" --replay-from "$root/models/COVER"
echo sampling > "$receipt/phase.txt"
"$edge" -m dunhuang.research.time_r29.evaluate --config "$cfg" --root "$root" --out "$root/evaluation_coverage" --domain aist --arm COVER
"$edge" -m dunhuang.research.time_r29.evaluate --config "$cfg" --root "$root" --out "$root/evaluation_coverage_replay" --domain aist --arm COVER --replay-from "$root/evaluation_coverage"
echo ChangE > "$receipt/phase.txt"
"$edge" -m dunhuang.research.time_r29.change --config "$cfg" --root "$root" --out "$root/change_coverage" --arms COVER
"$edge" -m dunhuang.research.time_r29.change --config "$cfg" --root "$root" --out "$root/change_coverage_replay" --arms COVER --replay-from "$root/change_coverage"
echo completed > "$receipt/phase.txt"
