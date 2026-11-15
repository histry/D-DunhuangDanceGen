#!/usr/bin/env bash
set -euo pipefail
root="${1:?fresh root}";receipt="${2:?fresh receipts}"
test ! -e "$root";test ! -e "$receipt";mkdir "$root" "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cfg=src/dunhuang/research/state_r27/config.json
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_state_r27.py > "$receipt/edge_tests.log" 2>&1
echo frozen_diagnostic > "$receipt/phase.txt"
"$edge" -m dunhuang.research.state_r27.diagnose --config "$cfg" --out "$root/diagnostic"
"$edge" -m dunhuang.research.state_r27.diagnose --config "$cfg" --out "$root/diagnostic_replay" --replay-from "$root/diagnostic"
echo teachers > "$receipt/phase.txt"
"$edge" -m dunhuang.research.state_r27.data --config "$cfg" --out "$root/teachers"
"$edge" -m dunhuang.research.state_r27.data --config "$cfg" --out "$root/teachers_replay" --replay-from "$root/teachers"
echo positive_training > "$receipt/phase.txt"
"$edge" -m dunhuang.research.state_r27.train --config "$cfg" --root "$root" --out "$root/models/P" --arm P
"$edge" -m dunhuang.research.state_r27.train --config "$cfg" --root "$root" --out "$root/models/P_replay" --arm P --replay-from "$root/models/P"
echo completed > "$receipt/phase.txt"
