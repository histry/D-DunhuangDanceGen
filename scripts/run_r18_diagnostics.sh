#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_diagnostic_receipts"
edge=/home/disk/lsm/conda_envs/edge/bin/python
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "${study}_receipts/exit_status.txt")" = 0
test ! -e "$receipt";mkdir "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
cfg=src/dunhuang/research/task_denoising_r18/config.json
"$edge" -m dunhuang.research.task_denoising_r18.diagnose --config "$cfg" --root "$study" --out "$study/objective_diagnostic"
"$edge" -m dunhuang.research.task_denoising_r18.diagnose --config "$cfg" --root "$study" --out "$study/objective_diagnostic_replay" --replay-from "$study/objective_diagnostic"
"$native" scripts/verify_r18.py --root "$study" --out "$study/independent_checks_after_sync.json"
