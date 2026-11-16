#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?receipt}"
test -f "$root/context/result.json";test -f "$root/context_replay/result.json";test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest tests/test_coverage_r34.py tests/test_plan_view_r34.py tests/test_transition_r35.py -q > "$receipt/edge_tests.log" 2>&1
echo correct_main_effect > "$receipt/phase.txt"
"$edge" -m dunhuang.research.transition_r35.generate --root "$root" --out "$root/generation" > "$receipt/generation.log" 2>&1
echo independent_main_effect_replay > "$receipt/phase.txt"
"$edge" -m dunhuang.research.transition_r35.generate --root "$root" --out "$root/generation_replay" --replay-from "$root/generation" > "$receipt/replay.log" 2>&1
echo viewer > "$receipt/phase.txt"
"$native" -m dunhuang.research.transition_r35.bundle --root "$root" --out "$root/viewer" > "$receipt/viewer.log" 2>&1
echo independent_verification > "$receipt/phase.txt"
"$native" scripts/release_r35.py --task verify --root "$root" > "$receipt/verification.log" 2>&1
echo completed > "$receipt/phase.txt"
