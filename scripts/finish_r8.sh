#!/usr/bin/env bash
set -euo pipefail
study="$1"
receipt="${study}_post_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH="$PWD/src"
test "$(cat "${study}_receipts/exit_status.txt")" = 0
test ! -e "$receipt"
mkdir "$receipt"
success=0
finish() { code=$?; if [ "$success" -ne 1 ] && [ "$code" -eq 0 ]; then code=125; fi; echo "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
date -u +%FT%TZ > "$receipt/started_utc.txt"
git rev-parse HEAD > "$receipt/execution_commit.txt"
cp "$study/independent_checks.json" "$receipt/previous_independent_checks.json"
"$native" -m pytest -q -rs tests/test_primitive_reperformance.py -k phase_proxy > "$receipt/phase_tests.log" 2>&1
"$native" -m dunhuang.research.primitive_reperformance.phase_analysis --single "$study/single" --out "$study/phase_proxy"
"$native" -m dunhuang.research.primitive_reperformance.phase_analysis --single "$study/single" --out "$study/phase_proxy_replay"
"$native" -m dunhuang.research.primitive_reperformance.render --config src/dunhuang/research/primitive_reperformance/config.json --diagnosis "$study/style_audit" --prepared "$study/prepared" --single "$study/single" --out "$study/videos_wrist_axes" --wrist-glyphs
"$native" scripts/verify_r8.py --out "$study" > "$receipt/independent_checks.log" 2>&1
"$native" -c 'import json,sys; from pathlib import Path; r=Path(sys.argv[1]); assert json.loads((r/"phase_proxy/result.json").read_text())==json.loads((r/"phase_proxy_replay/result.json").read_text())' "$study"
success=1
