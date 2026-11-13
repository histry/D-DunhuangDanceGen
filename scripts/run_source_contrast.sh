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
"$edge" -m pytest -q tests/test_source_contrast.py tests/test_source_expression.py tests/test_protected_source.py > "$receipt/edge_check.log" 2>&1
cfg=configs/source_contrast_r7_v1.json
"$native" -m dunhuang.research.source_contrast.preflight --config "$cfg" --out "$study/preflight"
"$native" -m dunhuang.research.source_contrast.preflight --config "$cfg" --out "$study/preflight_replay" --replay-from "$study/preflight"
"$edge" -m dunhuang.research.source_contrast.study --config "$cfg" --preflight "$study/preflight" --out "$study/control"
"$edge" -m dunhuang.research.source_contrast.study --config "$cfg" --preflight "$study/preflight" --out "$study/control_replay" --replay-from "$study/control"
success=1
