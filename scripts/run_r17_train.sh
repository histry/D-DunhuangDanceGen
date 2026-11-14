#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_train_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "${study}_audit_receipts/exit_status.txt")" = 0
test ! -e "$receipt";mkdir "$receipt";mkdir "$study/models"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
"$edge" -m pytest -q tests/test_global_clock_r17.py > "$receipt/edge_tests.log" 2>&1
cfg=src/dunhuang/research/global_clock_r17/config.json
"$native" -m dunhuang.research.global_clock_r17.evaluate --config "$cfg" --root "$study" --out "$study/audit_native" --audit-only
"$native" -m dunhuang.research.global_clock_r17.evaluate --config "$cfg" --root "$study" --out "$study/audit_native_replay" --audit-only --replay-from "$study/audit_native"
for mode in cross self;do
 "$edge" -m dunhuang.research.global_clock_r17.train --config "$cfg" --data "$study/data" --mode "$mode" --out "$study/models/$mode"
 "$edge" -m dunhuang.research.global_clock_r17.train --config "$cfg" --data "$study/data" --mode "$mode" --out "$study/models/${mode}_replay" --replay-from "$study/models/$mode"
done
"$edge" -m dunhuang.research.global_clock_r17.rollout --config "$cfg" --data "$study/data" --models "$study/models" --out "$study/rollout"
"$edge" -m dunhuang.research.global_clock_r17.rollout --config "$cfg" --data "$study/data" --models "$study/models" --out "$study/rollout_replay" --replay-from "$study/rollout"
