#!/usr/bin/env bash
set -euo pipefail
root="${1:?fresh root}";receipt="${2:?fresh receipts}"
test ! -e "$root";test ! -e "$receipt";mkdir "$root" "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/time_r29/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_time_r29.py > "$receipt/edge_tests.log" 2>&1
echo FineDance_windows > "$receipt/phase.txt"
"$edge" -m dunhuang.research.time_r29.data --config "$cfg" --out "$root/fine_data"
"$edge" -m dunhuang.research.time_r29.data --config "$cfg" --out "$root/fine_data_replay" --replay-from "$root/fine_data"
echo B0_parity > "$receipt/phase.txt"
"$edge" -m dunhuang.research.time_r29.parity --config "$cfg" --root "$root" --out "$root/parity"
"$edge" -m dunhuang.research.time_r29.parity --config "$cfg" --root "$root" --out "$root/parity_replay" --replay-from "$root/parity"
echo temporal_readouts > "$receipt/phase.txt"
"$edge" -m dunhuang.research.time_r29.readout --config "$cfg" --root "$root" --out "$root/readout"
"$edge" -m dunhuang.research.time_r29.readout --config "$cfg" --root "$root" --out "$root/readout_replay" --replay-from "$root/readout"
echo completed > "$receipt/phase.txt"
