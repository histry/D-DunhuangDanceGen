#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_tiny_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "${study}_audit_receipts/exit_status.txt")" = 0
test ! -e "$receipt";mkdir "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
cfg=src/dunhuang/research/masked_denoiser/config.json
"$edge" -m dunhuang.research.masked_denoiser.train --config "$cfg" --out "$study/tiny"
"$edge" -m dunhuang.research.masked_denoiser.train --config "$cfg" --out "$study/tiny_replay" --replay-from "$study/tiny"
"$edge" -m dunhuang.research.masked_denoiser.evaluate --config "$cfg" --training "$study/tiny" --out "$study/tiny_rollout"
"$edge" -m dunhuang.research.masked_denoiser.evaluate --config "$cfg" --training "$study/tiny" --out "$study/tiny_rollout_replay" --replay-from "$study/tiny_rollout"
"$native" -m dunhuang.research.masked_denoiser.analyze --config "$cfg" --source "$study/tiny_rollout" --training "$study/tiny" --out "$study/tiny_native" --phase tiny
"$native" -m dunhuang.research.masked_denoiser.analyze --config "$cfg" --source "$study/tiny_rollout_replay" --training "$study/tiny_replay" --out "$study/tiny_native_replay" --phase tiny --replay-from "$study/tiny_native"
