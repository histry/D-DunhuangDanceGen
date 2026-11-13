#!/usr/bin/env bash
set -euo pipefail
study="$1"
receipt="${study}_receipts"
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
"$edge" -m pytest -q -rs tests/test_primitive_reperformance.py tests/test_bridge_inpainting.py > "$receipt/edge_tests.log" 2>&1
s=src/dunhuang/research/primitive_reperformance/config.json
b=src/dunhuang/research/bridge_inpainting/config.json
"$native" -m dunhuang.research.primitive_reperformance.diagnosis --config "$s" --out "$study/style_audit"
"$native" -m dunhuang.research.bridge_inpainting.audit --config "$b" --out "$study/boundary_audit"
"$native" -m dunhuang.research.primitive_reperformance.prepare --config "$s" --out "$study/prepared"
"$native" -m dunhuang.research.primitive_reperformance.study --config "$s" --prepared "$study/prepared" --out "$study/single"
"$native" -m dunhuang.research.primitive_reperformance.study --config "$s" --prepared "$study/prepared" --out "$study/single_replay" --replay-from "$study/single"
"$native" -m dunhuang.research.bridge_inpainting.study --config "$b" --reperformance-config "$s" --audit "$study/boundary_audit" --out "$study/bridge"
"$native" -m dunhuang.research.bridge_inpainting.study --config "$b" --reperformance-config "$s" --audit "$study/boundary_audit" --out "$study/bridge_replay" --replay-from "$study/bridge"
"$native" -m dunhuang.research.primitive_reperformance.study --config "$s" --prepared "$study/prepared" --single "$study/single" --bridge "$study/bridge" --out "$study/whole"
"$native" -m dunhuang.research.primitive_reperformance.study --config "$s" --prepared "$study/prepared" --single "$study/single" --bridge "$study/bridge" --out "$study/whole_replay" --replay-from "$study/whole"
"$native" -m dunhuang.research.bridge_inpainting.whole --config "$b" --reperformance-config "$s" --whole "$study/whole" --out "$study/joint"
"$native" -m dunhuang.research.bridge_inpainting.whole --config "$b" --reperformance-config "$s" --whole "$study/whole" --out "$study/joint_replay" --replay-from "$study/joint"
"$native" -m dunhuang.research.primitive_reperformance.render --config "$s" --diagnosis "$study/style_audit" --prepared "$study/prepared" --single "$study/single" --out "$study/videos"
"$native" scripts/verify_r8.py --out "$study" > "$receipt/independent_checks.log" 2>&1
success=1
