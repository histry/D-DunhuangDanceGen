#!/usr/bin/env bash
set -euo pipefail
root="${1:?fresh root}";receipt="${2:?fresh receipts}"
test ! -e "$root";test ! -e "$receipt";mkdir "$root" "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/contract_r28/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_contract_r28.py > "$receipt/edge_tests.log" 2>&1
echo data > "$receipt/phase.txt"
"$edge" -m dunhuang.research.contract_r28.data --config "$cfg" --out "$root/data"
"$edge" -m dunhuang.research.contract_r28.data --config "$cfg" --out "$root/data_replay" --replay-from "$root/data"
echo factorial > "$receipt/phase.txt"
"$edge" -m dunhuang.research.contract_r28.factorial --config "$cfg" --root "$root" --out "$root/factorial"
"$edge" -m dunhuang.research.contract_r28.factorial --config "$cfg" --root "$root" --out "$root/factorial_replay" --replay-from "$root/factorial"
echo frozen_readout > "$receipt/phase.txt"
"$edge" -m dunhuang.research.contract_r28.readout --config "$cfg" --root "$root" --out "$root/readout"
"$edge" -m dunhuang.research.contract_r28.readout --config "$cfg" --root "$root" --out "$root/readout_replay" --replay-from "$root/readout"
echo completed > "$receipt/phase.txt"
