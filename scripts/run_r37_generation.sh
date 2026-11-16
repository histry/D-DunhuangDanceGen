#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";first_media="${2:?delivered viewer}";test -f "$first_media/protocol.json";test -f "$root/planning/song0_plan.json"
r="$root/receipts/generation";test ! -e "$r";mkdir -p "$r"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$r/code_commit.txt";sha256sum "$first_media/protocol.json" > "$r/first_media_protocol.txt";date -u +%FT%TZ > "$r/started_utc.txt"
trap 'code=$?;echo "$code" > "$r/exit_status.txt";date -u +%FT%TZ > "$r/finished_utc.txt";exit "$code"' EXIT
echo two_fixed_shared_boundary_cases > "$r/phase.txt"
"$py" -m dunhuang.research.transition_r37.generate --root "$root" --out "$root/generation" > "$r/generation.log" 2>&1
echo fresh_generation_replay > "$r/phase.txt"
"$py" -m dunhuang.research.transition_r37.generate --root "$root" --out "$root/generation_replay" --replay-from "$root/generation" > "$r/replay.log" 2>&1
echo completed > "$r/phase.txt"
