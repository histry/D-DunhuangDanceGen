#!/usr/bin/env bash
set -euo pipefail
study="$1"
receipt="$2"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
export PYTHONPATH="$PWD/src"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
mkdir -p "$study" "$receipt"
success=0
finish() { code=$?; if [ "$success" -ne 1 ] && [ "$code" -eq 0 ]; then code=125; fi; printf '%s\n' "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%FT%TZ > "$receipt/started_utc.txt"
git rev-parse HEAD > "$receipt/execution_commit.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_b2.py::test_trajectory_guidance_controls_fk_derivatives tests/test_source_identifiability.py > "$receipt/edge_check.log" 2>&1
"$native" -m dunhuang.research.source_identifiability.a2 --source outputs/dense_condition_03/a1 --config configs/content_identifiability_v2.json --out "$study/a2"
"$native" -m dunhuang.research.source_identifiability.a2 --source outputs/dense_condition_03/a1 --config configs/content_identifiability_v2.json --out "$study/a2_replay" --replay-from "$study/a2"
"$edge" -m dunhuang.research.source_identifiability.g2 --assets inputs/clock_assets_v2.json --root-protocol outputs/clock_root_02/root/protocol.json --config configs/g2_source_identifiability.json --out "$study/g2"
"$edge" -m dunhuang.research.source_identifiability.g2 --assets inputs/clock_assets_v2.json --root-protocol outputs/clock_root_02/root/protocol.json --config configs/g2_source_identifiability.json --out "$study/g2_replay" --replay-from "$study/g2"
"$native" scripts/export_source_identifiability.py --out "$study" --receipt "$receipt"
success=1
