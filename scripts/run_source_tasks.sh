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
"$edge" -m pytest -q tests/test_b2.py::test_trajectory_guidance_controls_fk_derivatives tests/test_source_tasks.py > "$receipt/edge_check.log" 2>&1
for suffix in main replay; do
  extra=()
  if [ "$suffix" = replay ]; then extra=(--replay-from "$study/effect_main"); fi
  "$native" -m dunhuang.research.source_tasks.source_effect --config configs/source_tasks_v1.json --out "$study/effect_$suffix" "${extra[@]}"
done
"$edge" -m dunhuang.research.source_tasks.qualify --config configs/source_tasks_v1.json --out "$study/pairs"
"$edge" -m dunhuang.research.source_tasks.qualify --config configs/source_tasks_v1.json --out "$study/pairs_replay" --replay-from "$study/pairs"
"$native" -m dunhuang.research.source_tasks.content_data --config configs/source_tasks_v1.json --out "$study/content_data"
"$native" -m dunhuang.research.source_tasks.content_data --config configs/source_tasks_v1.json --out "$study/content_data_replay" --replay-from "$study/content_data"
"$edge" -m dunhuang.research.source_tasks.cf --config configs/source_tasks_v1.json --qualified "$study/pairs" --out "$study/cf"
"$edge" -m dunhuang.research.source_tasks.cf --config configs/source_tasks_v1.json --qualified "$study/pairs" --out "$study/cf_replay" --replay-from "$study/cf"
"$edge" -m dunhuang.research.source_tasks.content --config configs/source_tasks_v1.json --prepared "$study/content_data" --out "$study/content"
"$edge" -m dunhuang.research.source_tasks.content --config configs/source_tasks_v1.json --prepared "$study/content_data" --out "$study/content_replay" --replay-from "$study/content"
"$native" scripts/export_source_tasks.py --out "$study" --receipt "$receipt"
success=1
