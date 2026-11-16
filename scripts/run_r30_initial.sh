#!/usr/bin/env bash
set -euo pipefail
root="${1:?fresh root}";receipt="${2:?fresh receipt}"
test ! -e "$root";test ! -e "$receipt";mkdir -p "$root" "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_continuity_r30.py > "$receipt/edge_tests.log" 2>&1
echo audit > "$receipt/phase.txt"
cfg=src/dunhuang/research/continuity_r30/config.json
"$edge" -m dunhuang.research.continuity_r30.audit --config "$cfg" --out "$root/audit"
"$edge" -m dunhuang.research.continuity_r30.audit --config "$cfg" --out "$root/audit_replay" --replay-from "$root/audit"
echo completed > "$receipt/phase.txt"
