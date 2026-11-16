#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test "$(cat ${root}_qualification_recovery_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/geometry_r31/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_geometry_r31.py > "$receipt/edge_tests.log" 2>&1
for arm in LegacyIO CanonicalIO;do
 echo "training_$arm" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.geometry_r31.train --config "$cfg" --root "$root" --out "$root/models/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.geometry_r31.train --config "$cfg" --root "$root" --out "$root/models/${arm}_replay" --arm "$arm" --replay-from "$root/models/$arm"
done
echo completed > "$receipt/phase.txt"
