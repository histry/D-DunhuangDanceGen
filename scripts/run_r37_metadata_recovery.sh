#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";r="$root/receipts/replay_metadata_recovery";test ! -e "$r";mkdir -p "$r"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cpu=/home/disk/lsm/conda_envs/dance-rag/bin/python
git rev-parse HEAD > "$r/code_commit.txt";date -u +%FT%TZ > "$r/started_utc.txt"
trap 'code=$?;echo "$code" > "$r/exit_status.txt";date -u +%FT%TZ > "$r/finished_utc.txt";exit "$code"' EXIT
echo waiting_for_full_fresh_search > "$r/phase.txt"
while ! test -f "$root/receipts/resume/exit_status.txt";do sleep 3;done
test -f "$root/planning_replay/result.json"
echo strict_numeric_and_scientific_blob_binding > "$r/phase.txt"
"$cpu" - "$root" > "$r/binding.log" 2>&1 <<'PY'
import sys
from pathlib import Path
from dunhuang.research.transition_r37.replay_contract import compare_planning
r=Path(sys.argv[1]);print(compare_planning(r/'planning',r/'planning_replay',r/'planning_replay_binding.json'))
PY
if ! test -f "$root/smoothing/result.json";then
  "$cpu" -m dunhuang.research.transition_r37.postprocess --out "$root/smoothing" > "$r/smoothing.log" 2>&1
  "$cpu" -m dunhuang.research.transition_r37.postprocess --out "$root/smoothing_replay" --replay-from "$root/smoothing" > "$r/smoothing_replay.log" 2>&1
fi
if ! test -f "$root/viewer_first/protocol.json";then
  "$cpu" -m dunhuang.research.transition_r37.view --root "$root" --out "$root/viewer_first" > "$r/viewer.log" 2>&1
fi
echo completed > "$r/phase.txt"
