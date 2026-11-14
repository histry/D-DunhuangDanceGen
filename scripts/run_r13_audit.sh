#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_audit_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test ! -e "$study/teacher";test ! -e "$receipt";mkdir -p "$study" "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_masked_denoiser.py > "$receipt/edge_tests.log" 2>&1
cfg=src/dunhuang/research/masked_denoiser/config.json
"$native" -m dunhuang.research.masked_denoiser.official --source "$study/official_source" --out "$study/official"
"$native" -m dunhuang.research.masked_denoiser.official --source "$study/official_source" --out "$study/official_replay" --replay-from "$study/official"
"$edge" -m dunhuang.research.masked_denoiser.teacher --config "$cfg" --out "$study/teacher"
"$edge" -m dunhuang.research.masked_denoiser.teacher --config "$cfg" --out "$study/teacher_replay" --replay-from "$study/teacher"
"$native" -m dunhuang.research.masked_denoiser.analyze --config "$cfg" --source "$study/teacher" --out "$study/teacher_native" --phase teacher
"$native" -m dunhuang.research.masked_denoiser.analyze --config "$cfg" --source "$study/teacher_replay" --out "$study/teacher_native_replay" --phase teacher --replay-from "$study/teacher_native"
