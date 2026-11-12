#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
study="$1"
receipt="$2"
source="$3"
clock="$4"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
python_bin=/home/disk/lsm/conda_envs/dance-rag/bin/python
mkdir -p "$receipt"
success=0
finish() { code=$?; if [ "$success" -ne 1 ] && [ "$code" -eq 0 ]; then code=125; fi; printf '%s\n' "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%FT%TZ > "$receipt/started_utc.txt"
git rev-parse HEAD > "$receipt/execution_commit.txt"
"$python_bin" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
/home/disk/lsm/conda_envs/edge/bin/python -m pytest -q tests/test_b2.py::test_trajectory_guidance_controls_fk_derivatives > "$receipt/edge_check.log" 2>&1
"$python_bin" -m dunhuang.alignment.local_identifiability --phase preflight --source "$source" --clock "$clock" --config configs/local_identifiability.json --out "$study"
"$python_bin" -m dunhuang.alignment.local_identifiability --phase run --out "$study"
"$python_bin" -m dunhuang.alignment.local_identifiability --phase replay --out "$study" --replay-out "$study/replay"
"$python_bin" scripts/export_local_identifiability.py --out "$study" --receipt "$receipt" --dest "$study/public_summary.json"
success=1
