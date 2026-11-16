#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test "$(cat ${root}_training_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/geometry_r31/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo actual_sampling > "$receipt/phase.txt"
"$edge" -m dunhuang.research.geometry_r31.evaluate --config "$cfg" --root "$root" --out "$root/evaluation"
"$edge" -m dunhuang.research.geometry_r31.evaluate --config "$cfg" --root "$root" --out "$root/evaluation_replay" --replay-from "$root/evaluation"
echo gradient_diagnosis > "$receipt/phase.txt"
"$edge" -m dunhuang.research.geometry_r31.probe --config "$cfg" --root "$root" --out "$root/probe"
"$edge" -m dunhuang.research.geometry_r31.probe --config "$cfg" --root "$root" --out "$root/probe_replay" --replay-from "$root/probe"
echo ChangE > "$receipt/phase.txt"
"$edge" -m dunhuang.research.geometry_r31.change --config "$cfg" --root "$root" --out "$root/change"
"$edge" -m dunhuang.research.geometry_r31.change --config "$cfg" --root "$root" --out "$root/change_replay" --replay-from "$root/change"
echo completed > "$receipt/phase.txt"
