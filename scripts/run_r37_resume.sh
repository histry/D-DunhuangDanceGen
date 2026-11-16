#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";r="$root/receipts/resume";test ! -e "$r";mkdir -p "$r"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py=/home/disk/lsm/conda_envs/dance-rag/bin/python
git rev-parse HEAD > "$r/code_commit.txt";date -u +%FT%TZ > "$r/started_utc.txt"
trap 'code=$?;echo "$code" > "$r/exit_status.txt";date -u +%FT%TZ > "$r/finished_utc.txt";exit "$code"' EXIT
echo parallel_independent_songs > "$r/phase.txt"
"$py" -m dunhuang.research.transition_r37.study --out "$root/planning" --resume-incomplete-plans > "$r/planning.log" 2>&1
echo full_fresh_parallel_replay > "$r/phase.txt"
"$py" -m dunhuang.research.transition_r37.study --out "$root/planning_replay" --replay-from "$root/planning" > "$r/planning_replay.log" 2>&1
echo fixed_postprocessing > "$r/phase.txt"
"$py" -m dunhuang.research.transition_r37.postprocess --out "$root/smoothing" > "$r/smoothing.log" 2>&1
"$py" -m dunhuang.research.transition_r37.postprocess --out "$root/smoothing_replay" --replay-from "$root/smoothing" > "$r/smoothing_replay.log" 2>&1
"$py" -m dunhuang.research.transition_r37.view --root "$root" --out "$root/viewer_first" > "$r/viewer.log" 2>&1
echo completed > "$r/phase.txt"
