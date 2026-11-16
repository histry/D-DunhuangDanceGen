#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";r="$root/receipts/finalize";test ! -e "$r";mkdir -p "$r"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cpu=/home/disk/lsm/conda_envs/dance-rag/bin/python
git rev-parse HEAD > "$r/code_commit.txt";date -u +%FT%TZ > "$r/started_utc.txt"
trap 'code=$?;echo "$code" > "$r/exit_status.txt";date -u +%FT%TZ > "$r/finished_utc.txt";exit "$code"' EXIT
echo independent_verification > "$r/phase.txt"
"$cpu" -m dunhuang.research.transition_r37.verify --root "$root" --out "$root/verification.json" > "$r/verify.log" 2>&1
"$cpu" -m dunhuang.research.transition_r37.content_diagnostics --root "$root" --out "$root/content_diagnostics.json" > "$r/content.log" 2>&1
echo final_watchable_effect > "$r/phase.txt"
"$cpu" -m dunhuang.research.transition_r37.view --root "$root" --generated "$root/generation" --out "$root/viewer_final" > "$r/viewer.log" 2>&1
echo regression > "$r/phase.txt"
"$cpu" -m pytest -q > "$r/regression.log" 2>&1
echo completed > "$r/phase.txt"
