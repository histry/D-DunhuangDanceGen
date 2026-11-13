#!/usr/bin/env bash
set -euo pipefail
study="$1"
receipt="$2"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH="$PWD/src"
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
"$edge" -m pytest -q tests/test_reference_guidance.py tests/test_source_capacity.py tests/test_b2.py::test_trajectory_guidance_controls_fk_derivatives > "$receipt/edge_check.log" 2>&1
"$edge" -m dunhuang.research.reference_guidance.events --config configs/reference_event_readout_v1.json --out "$study/events"
"$edge" -m dunhuang.research.reference_guidance.events --config configs/reference_event_readout_v1.json --out "$study/events_replay" --replay-from "$study/events"
"$native" scripts/export_reference_guidance.py --out "$study" --receipt "$receipt" --control-root outputs/reference_guidance_03
success=1
