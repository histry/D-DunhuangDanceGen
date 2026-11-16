#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}";test -f "$root/context/context.pt";test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python;edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo budget_contract_regression > "$receipt/phase.txt"
"$native" -m pytest tests/test_complete_r36.py -q > "$receipt/tests.log" 2>&1
echo same_weights_same_seeds_edit_repair > "$receipt/phase.txt"
"$edge" -m dunhuang.research.complete_r36.generate --root "$root" --out "$root/generation_repair01" > "$receipt/generation.log" 2>&1
echo independent_replay > "$receipt/phase.txt"
"$edge" -m dunhuang.research.complete_r36.generate --root "$root" --out "$root/generation_repair_replay01" --replay-from "$root/generation_repair01" > "$receipt/replay.log" 2>&1
echo independent_verification > "$receipt/phase.txt"
"$native" -m dunhuang.research.complete_r36.verify --root "$root" --generation generation_repair01 --generation-replay generation_repair_replay01 --out "$root/independent_checks_repair.json" > "$receipt/verify.log" 2>&1
echo watchable_repair > "$receipt/phase.txt"
"$native" -m dunhuang.research.complete_r36.view --root "$root" --generated "$root/generation_repair01" --out "$root/viewer_final03" > "$receipt/viewer.log" 2>&1
echo evidence_release > "$receipt/phase.txt"
"$native" -m dunhuang.research.complete_r36.release --root "$root" --generation generation_repair01 --checks independent_checks_repair.json --public-dir public_repair --manifest artifacts_manifest_repair.json > "$receipt/release.log" 2>&1
echo completed > "$receipt/phase.txt"
