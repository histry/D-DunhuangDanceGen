#!/usr/bin/env bash
set -euo pipefail
root="${1:?fresh root}";receipt="${2:?fresh receipt}"
test ! -e "$root";test ! -e "$receipt";mkdir -p "$root";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/coverage_r34/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo tests > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest tests/test_coverage_r34.py -q > "$receipt/edge_tests.log" 2>&1
echo continuous_observations_and_parity > "$receipt/phase.txt"
"$edge" -m dunhuang.research.coverage_r34.prepare --config "$cfg" --out "$root/data"
echo fresh_replay > "$receipt/phase.txt"
"$edge" -m dunhuang.research.coverage_r34.prepare --config "$cfg" --out "$root/data_replay" --replay-from "$root/data"
echo completed > "$receipt/phase.txt"
