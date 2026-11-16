#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test "$(cat ${root}_initial_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/continuity_r30/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_continuity_r30.py > "$receipt/edge_tests.log" 2>&1
echo reading_training > "$receipt/phase.txt"
"$edge" -m dunhuang.research.continuity_r30.train --config "$cfg" --root "$root" --out "$root/models/R1" --arm R1
"$edge" -m dunhuang.research.continuity_r30.train --config "$cfg" --root "$root" --out "$root/models/R1_replay" --arm R1 --replay-from "$root/models/R1"
echo Train_only_evaluation > "$receipt/phase.txt"
"$edge" -m dunhuang.research.continuity_r30.evaluate --config "$cfg" --root "$root" --out "$root/reading_train" --arms R1 --roles train
"$edge" -m dunhuang.research.continuity_r30.evaluate --config "$cfg" --root "$root" --out "$root/reading_train_replay" --arms R1 --roles train --replay-from "$root/reading_train"
echo completed > "$receipt/phase.txt"
