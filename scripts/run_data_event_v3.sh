#!/usr/bin/env bash
set -euo pipefail
study="$1"
receipt="$2"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
export PYTHONPATH="$PWD/src"
test ! -e "$study"
test ! -e "$receipt"
mkdir -p "$study" "$receipt"
success=0
finish() { code=$?; if [ "$success" -ne 1 ] && [ "$code" -eq 0 ]; then code=125; fi; printf '%s\n' "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%FT%TZ > "$receipt/started_utc.txt"
git rev-parse HEAD > "$receipt/execution_commit.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_b2.py::test_trajectory_guidance_controls_fk_derivatives tests/test_data_event_v3.py > "$receipt/edge_check.log" 2>&1
"$native" -m dunhuang.research.data_event_v3.study --assets inputs/clock_assets_v2.json --population assets/datasets/population.json --config configs/data_event_v3.json --out "$study/main"
"$native" -m dunhuang.research.data_event_v3.study --assets inputs/clock_assets_v2.json --population assets/datasets/population.json --config configs/data_event_v3.json --out "$study/replay" --replay-from "$study/main"
"$native" scripts/export_data_event_v3.py --out "$study" --receipt "$receipt"
success=1
