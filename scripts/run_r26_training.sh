#!/usr/bin/env bash
set -euo pipefail
study="${1:?fresh study}";receipt="${2:?fresh receipts}"
test ! -e "$study";test ! -e "$receipt";mkdir -p "$study" "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cfg=src/dunhuang/research/reference_r26/config.json
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_reference_r26.py > "$receipt/edge_tests.log" 2>&1
echo data > "$receipt/phase.txt"
"$edge" -m dunhuang.research.reference_r26.data --config "$cfg" --out "$study/data"
"$edge" -m dunhuang.research.reference_r26.data --config "$cfg" --out "$study/data_replay" --replay-from "$study/data"
echo preflight > "$receipt/phase.txt"
"$edge" -m dunhuang.research.reference_r26.preflight --config "$cfg" --root "$study" --out "$study/preflight"
"$edge" -m dunhuang.research.reference_r26.preflight --config "$cfg" --root "$study" --out "$study/preflight_replay" --replay-from "$study/preflight"
for arm in C0 C1 C2;do
 echo "training_$arm" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.reference_r26.train --config "$cfg" --root "$study" --out "$study/models/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.reference_r26.train --config "$cfg" --root "$study" --out "$study/models/${arm}_replay" --arm "$arm" --replay-from "$study/models/$arm"
done
echo completed > "$receipt/phase.txt"
