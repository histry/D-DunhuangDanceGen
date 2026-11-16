#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test "$(cat ${root}_preview_receipts/exit_status.txt)" = 0
test -f "$root/plan_generation_replay/result.json"
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest tests/test_coverage_r34.py tests/test_plan_view_r34.py -q > "$receipt/edge_tests.log" 2>&1
echo independent_pcm_feature_motion_verification > "$receipt/phase.txt"
"$native" -m dunhuang.research.plan_view_r34.verify --root "$root" --out "$root/plan_generation_validation"
echo display_bundle > "$receipt/phase.txt"
"$native" -m dunhuang.research.plan_view_r34.bundle --root "$root" --out "$root/viewer"
echo completed > "$receipt/phase.txt"
