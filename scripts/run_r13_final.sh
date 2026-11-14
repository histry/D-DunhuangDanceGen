#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_final_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "${study}_tiny_receipts/exit_status.txt")" = 0
test ! -e "$receipt";mkdir "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_masked_denoiser.py > "$receipt/edge_tests.log" 2>&1
"$native" scripts/verify_r13.py --root "$study"
