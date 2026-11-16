#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}";test -f "$root/planning/result.json";test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python;edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q > "$receipt/tests.log" 2>&1
echo main_context_correction > "$receipt/phase.txt"
"$edge" -m dunhuang.research.complete_r36.train --root "$root" --out "$root/context" > "$receipt/train.log" 2>&1
echo independent_training_readouts > "$receipt/phase.txt"
"$edge" -m dunhuang.research.complete_r36.train --root "$root" --out "$root/context_replay" --replay-from "$root/context" > "$receipt/train_replay.log" 2>&1
echo generation > "$receipt/phase.txt"
"$edge" -m dunhuang.research.complete_r36.generate --root "$root" --out "$root/generation" > "$receipt/generation.log" 2>&1
echo independent_generation_replay > "$receipt/phase.txt"
"$edge" -m dunhuang.research.complete_r36.generate --root "$root" --out "$root/generation_replay" --replay-from "$root/generation" > "$receipt/replay.log" 2>&1
echo final_viewer > "$receipt/phase.txt"
"$native" -m dunhuang.research.complete_r36.view --root "$root" --generated "$root/generation" --out "$root/viewer" > "$receipt/viewer.log" 2>&1
echo independent_verification > "$receipt/phase.txt"
"$native" -m dunhuang.research.complete_r36.verify --root "$root" --out "$root/independent_checks.json" > "$receipt/verify.log" 2>&1
echo completed > "$receipt/phase.txt"
