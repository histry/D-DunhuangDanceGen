#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test ! -e "$root";test ! -e "$receipt";mkdir -p "$root";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/content_r32/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo tests > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest tests/test_content_r32.py -q > "$receipt/edge_tests.log" 2>&1
echo platform_parity > "$receipt/phase.txt"
"$edge" -m dunhuang.research.content_r32.prepare --config "$cfg" --out "$root/data"
"$edge" -m dunhuang.research.content_r32.prepare --config "$cfg" --out "$root/data_replay" --replay-from "$root/data"
echo readout > "$receipt/phase.txt"
"$edge" -m dunhuang.research.content_r32.readout --config "$cfg" --out "$root/readout"
"$edge" -m dunhuang.research.content_r32.readout --config "$cfg" --out "$root/readout_replay" --replay-from "$root/readout"
echo completed > "$receipt/phase.txt"
