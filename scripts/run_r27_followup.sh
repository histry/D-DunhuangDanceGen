#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipts}"
test "$(cat ${root}_initial_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cfg=src/dunhuang/research/state_r27/config.json
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo positive_evaluation > "$receipt/phase.txt"
"$edge" -m dunhuang.research.state_r27.evaluate --config "$cfg" --root "$root" --out "$root/positive" --models R26_C1 P
"$edge" -m dunhuang.research.state_r27.evaluate --config "$cfg" --root "$root" --out "$root/positive_replay" --models R26_C1 P --replay-from "$root/positive"
"$native" -m dunhuang.research.state_r27.route --config "$cfg" --root "$root"
"$native" -m dunhuang.research.state_r27.branch_summary --root "$root"
echo completed > "$receipt/phase.txt"
