#!/usr/bin/env bash
set -euo pipefail
study="$1"
receipt="$2"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
mkdir -p "$receipt" "$study"
success=0
finish() { code=$?; if [ "$success" -ne 1 ] && [ "$code" -eq 0 ]; then code=125; fi; printf '%s\n' "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%FT%TZ > "$receipt/started_utc.txt"
git rev-parse HEAD > "$receipt/execution_commit.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_b2.py::test_trajectory_guidance_controls_fk_derivatives tests/test_dense_condition_audit.py > "$receipt/edge_check.log" 2>&1
"$edge" scripts/event_causal_audit.py --assets inputs/clock_assets_v2.json --root-protocol outputs/clock_root_02/root/protocol.json --config configs/event_causal_audit.json --out "$study/g1"
"$edge" scripts/event_causal_audit.py --assets inputs/clock_assets_v2.json --root-protocol outputs/clock_root_02/root/protocol.json --config configs/event_causal_audit.json --out "$study/g1_replay" --replay-from "$study/g1"
"$native" scripts/dense_event_experiment.py --source outputs/event_id_02 --clock outputs/clock_root_04/alignment --config configs/dense_event_coverage.json --out "$study/a1"
"$native" scripts/dense_event_experiment.py --source outputs/event_id_02 --clock outputs/clock_root_04/alignment --config configs/dense_event_coverage.json --out "$study/a1_replay" --replay-from "$study/a1"
"$native" scripts/export_dense_condition_audit.py --out "$study" --receipt "$receipt" --dest "$study/public_summary.json"
success=1
