#!/usr/bin/env bash
set -euo pipefail
study="$1"
receipt="$study"_receipts
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test ! -e "$study"; test ! -e "$receipt"; mkdir -p "$study" "$receipt"
finish(){ code=$?; echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
p=src/dunhuang/research/condition_dependency/config.json
r=src/dunhuang/research/root_foot/config.json
h=src/dunhuang/research/boundary_reference/config.json
"$edge" -m dunhuang.research.condition_dependency.diagnose --config "$p" --out "$study/p0"
"$edge" -m dunhuang.research.condition_dependency.diagnose --config "$p" --out "$study/p0_replay" --replay-from "$study/p0"
"$native" -m dunhuang.research.root_foot.capacity --config "$r" --out "$study/root"
"$native" -m dunhuang.research.root_foot.capacity --config "$r" --out "$study/root_replay" --replay-from "$study/root"
"$native" -m dunhuang.research.boundary_reference.study --config "$h" --out "$study/hermite"
"$native" -m dunhuang.research.boundary_reference.study --config "$h" --out "$study/hermite_replay" --replay-from "$study/hermite"
