#!/usr/bin/env bash
set -euo pipefail
root="${1:?fresh preview root}";test ! -e "$root";mkdir -p "$root/receipts"
r="$root/receipts";export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cpu=/home/disk/lsm/conda_envs/dance-rag/bin/python
git rev-parse HEAD > "$r/code_commit.txt"
trap 'code=$?;echo "$code" > "$r/exit_status.txt";exit "$code"' EXIT
"$cpu" -m dunhuang.research.transition_r37.postprocess --out "$root/smoothing" > "$r/smoothing.log" 2>&1
"$cpu" -m dunhuang.research.transition_r37.postprocess --out "$root/smoothing_replay" --replay-from "$root/smoothing" > "$r/replay.log" 2>&1
"$cpu" -m dunhuang.research.transition_r37.view --root "$root" --out "$root/viewer" --source-and-smoothing-only > "$r/viewer.log" 2>&1
