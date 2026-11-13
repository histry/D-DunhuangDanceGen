#!/usr/bin/env bash
set -euo pipefail
parent="$1"
study="$2"
receipt="$study"_receipts
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "$parent"_receipts/exit_status.txt)" = 0
test ! -e "$study"; test ! -e "$receipt"; mkdir -p "$study" "$receipt"
finish() { code=$?; echo "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%FT%TZ > "$receipt/started_utc.txt"
git rev-parse HEAD > "$receipt/execution_commit.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q -rs tests/test_boundary_context.py > "$receipt/edge_tests.log" 2>&1
"$native" -m dunhuang.research.boundary_state.preflight --out "$study/data_replay"
"$native" -c 'import json,sys;from pathlib import Path;a,b=map(Path,sys.argv[1:]);assert json.loads((a/"training_preflight/result.json").read_text())==json.loads((b/"data_replay/result.json").read_text());assert json.loads((a/"training_preflight/rows.json").read_text())==json.loads((b/"data_replay/rows.json").read_text());assert json.loads((b/"data_replay/result.json").read_text())["bridge_training_data_ready"];assert json.loads((a/"boundary/result.json").read_text())["formal_passes"]==0' "$parent" "$study"
cfg=src/dunhuang/research/boundary_context/config.json
"$edge" -m dunhuang.research.boundary_context.train --config "$cfg" --preflight "$parent/training_preflight" --out "$study/train"
"$edge" -m dunhuang.research.boundary_context.train --config "$cfg" --preflight "$parent/training_preflight" --out "$study/train_replay" --replay-from "$study/train"
"$native" -m dunhuang.research.boundary_context.study --config "$cfg" --preflight "$parent/training_preflight" --training "$study/train" --out "$study/generation"
"$native" -m dunhuang.research.boundary_context.study --config "$cfg" --preflight "$parent/training_preflight" --training "$study/train" --out "$study/generation_replay" --replay-from "$study/generation"
"$native" -m dunhuang.research.boundary_context.application --parent outputs/reperformance_r8_02 --training "$study/train" --out "$study/application"
"$native" -m dunhuang.research.boundary_context.application --parent outputs/reperformance_r8_02 --training "$study/train" --out "$study/application_replay" --replay-from "$study/application"
"$native" scripts/verify_b2.py --out "$study" > "$receipt/independent_checks.log" 2>&1
