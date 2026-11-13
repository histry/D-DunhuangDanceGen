#!/usr/bin/env bash
set -euo pipefail
study="$1"
receipt="$2"
python=/home/disk/lsm/conda_envs/dance-rag/bin/python
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH="$PWD/src"
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
"$python" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
source=outputs/source_expression_04/review/reviewer
catalog=outputs/event_plan_01/prepare/catalog/catalog.json
"$python" -m dunhuang.research.source_expression.viewer_v2 --source "$source" --catalog "$catalog" --out "$study/main"
"$python" -m dunhuang.research.source_expression.viewer_v2 --source "$source" --catalog "$catalog" --out "$study/replay" --replay-from "$study/main"
"$python" scripts/check_review_viewer.py --study "$study" --source "$source"
success=1
