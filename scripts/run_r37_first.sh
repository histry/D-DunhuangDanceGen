#!/usr/bin/env bash
set -euo pipefail
root="${1:?fresh root}";test ! -e "$root";mkdir -p "$root/receipts/first"
r="$root/receipts/first";export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py=/home/disk/lsm/conda_envs/dance-rag/bin/python
git rev-parse HEAD > "$r/code_commit.txt";date -u +%FT%TZ > "$r/started_utc.txt"
trap 'code=$?;echo "$code" > "$r/exit_status.txt";date -u +%FT%TZ > "$r/finished_utc.txt";exit "$code"' EXIT
echo contracts > "$r/phase.txt"
"$py" -m pytest tests/test_transition_r37.py -q > "$r/tests.log" 2>&1
echo fixed_catalog_planning > "$r/phase.txt"
"$py" -m dunhuang.research.transition_r37.study --out "$root/planning" > "$r/planning.log" 2>&1
"$py" -m dunhuang.research.transition_r37.study --out "$root/planning_replay" --replay-from "$root/planning" > "$r/planning_replay.log" 2>&1
echo fixed_postprocessing > "$r/phase.txt"
"$py" -m dunhuang.research.transition_r37.postprocess --out "$root/smoothing" > "$r/smoothing.log" 2>&1
"$py" -m dunhuang.research.transition_r37.postprocess --out "$root/smoothing_replay" --replay-from "$root/smoothing" > "$r/smoothing_replay.log" 2>&1
echo first_watchable_effect > "$r/phase.txt"
"$py" -m dunhuang.research.transition_r37.view --root "$root" --out "$root/viewer_first" > "$r/viewer.log" 2>&1
echo completed > "$r/phase.txt"
