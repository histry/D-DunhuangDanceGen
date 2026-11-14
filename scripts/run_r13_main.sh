#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_main_receipts"
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
cfg=src/dunhuang/research/masked_denoiser/config.json
"$edge" -m dunhuang.research.masked_denoiser.train --config "$cfg" --stage main --gate "$study/tiny_native" --out "$study/main"
"$edge" -m dunhuang.research.masked_denoiser.train --config "$cfg" --stage main --gate "$study/tiny_native" --out "$study/main_replay" --replay-from "$study/main"
"$edge" -m dunhuang.research.masked_denoiser.evaluate --config "$cfg" --stage main --training "$study/main" --out "$study/main_rollout"
"$edge" -m dunhuang.research.masked_denoiser.evaluate --config "$cfg" --stage main --training "$study/main" --out "$study/main_rollout_replay" --replay-from "$study/main_rollout"
"$native" -m dunhuang.research.masked_denoiser.analyze --config "$cfg" --source "$study/main_rollout" --training "$study/main" --out "$study/main_native" --phase main
"$native" -m dunhuang.research.masked_denoiser.analyze --config "$cfg" --source "$study/main_rollout_replay" --training "$study/main_replay" --out "$study/main_native_replay" --phase main --replay-from "$study/main_native"
