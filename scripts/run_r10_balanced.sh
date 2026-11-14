#!/usr/bin/env bash
set -euo pipefail
study="$1"
receipt="$study"_receipts
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat outputs/content_r10_01_receipts/exit_status.txt)" = 0
test ! -e "$study";test ! -e "$receipt";mkdir -p "$study" "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_r10_condition.py tests/test_r10_capacity.py > "$receipt/edge_tests.log" 2>&1
cfg=src/dunhuang/research/condition_dependency/balanced.json
"$edge" -m dunhuang.research.condition_dependency.units --config "$cfg" --out "$study/units"
"$edge" -m dunhuang.research.condition_dependency.units --config "$cfg" --out "$study/units_replay" --replay-from "$study/units"
"$edge" -m dunhuang.research.condition_dependency.train --config "$cfg" --audit outputs/dependency_r10_01 --out "$study/C2" --arm C2
"$edge" -m dunhuang.research.condition_dependency.train --config "$cfg" --audit outputs/dependency_r10_01 --out "$study/C2_replay" --arm C2 --replay-from "$study/C2"
"$native" -m dunhuang.research.condition_dependency.generate --config "$cfg" --training "$study" --out "$study/generation"
"$native" -m dunhuang.research.condition_dependency.generate --config "$cfg" --training "$study" --out "$study/generation_replay" --replay-from "$study/generation"
"$native" scripts/verify_r10_balanced.py --root "$study" > "$receipt/independent_checks.log" 2>&1
