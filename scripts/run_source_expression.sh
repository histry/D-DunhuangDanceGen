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
"$edge" -m pytest -q tests/test_source_expression.py tests/test_protected_source.py tests/test_reference_guidance.py tests/test_source_capacity.py tests/test_b2.py::test_trajectory_guidance_controls_fk_derivatives > "$receipt/edge_check.log" 2>&1
cfg=configs/source_expression_r6_v1.json
"$edge" -m dunhuang.research.source_expression.prepare --config "$cfg" --out "$study/prepared"
"$edge" -m dunhuang.research.source_expression.prepare --config "$cfg" --out "$study/prepared_replay" --replay-from "$study/prepared"
"$edge" -m dunhuang.research.source_expression.diagnose --config "$cfg" --out "$study/diagnosis"
"$edge" -m dunhuang.research.source_expression.diagnose --config "$cfg" --out "$study/diagnosis_replay" --replay-from "$study/diagnosis"
"$edge" -m dunhuang.research.source_expression.probe --config "$cfg" --prepared "$study/prepared" --diagnosis "$study/diagnosis" --out "$study/probe"
"$edge" -m dunhuang.research.source_expression.probe --config "$cfg" --prepared "$study/prepared" --diagnosis "$study/diagnosis" --out "$study/probe_replay" --replay-from "$study/probe"
for stage in music review; do
"$native" -m "dunhuang.research.source_expression.$stage" --config "$cfg" --prepared "$study/prepared" --out "$study/$stage"
"$native" -m "dunhuang.research.source_expression.$stage" --config "$cfg" --prepared "$study/prepared" --out "$study/${stage}_replay" --replay-from "$study/$stage"
done
"$native" scripts/export_source_expression.py --out "$study" --receipt "$receipt"
success=1
