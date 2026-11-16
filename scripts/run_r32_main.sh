#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test "$(cat ${root}_pilot_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/content_r32/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo tests > "$receipt/phase.txt"
"$edge" -m pytest tests/test_content_r32.py -q > "$receipt/edge_tests.log" 2>&1
for arm in AIST ChangE;do
 echo "training_$arm" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.content_r32.train --config "$cfg" --root "$root" --out "$root/models/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.content_r32.train --config "$cfg" --root "$root" --out "$root/models/${arm}_replay" --arm "$arm" --replay-from "$root/models/$arm"
done
echo evaluation > "$receipt/phase.txt"
"$edge" -m dunhuang.research.content_r32.evaluate --config "$cfg" --root "$root" --out "$root/evaluation"
"$edge" -m dunhuang.research.content_r32.evaluate --config "$cfg" --root "$root" --out "$root/evaluation_replay" --replay-from "$root/evaluation"
echo completed > "$receipt/phase.txt"
