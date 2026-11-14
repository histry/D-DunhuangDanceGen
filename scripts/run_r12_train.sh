#!/usr/bin/env bash
set -euo pipefail
study="$1"; receipt="${study}_train_receipts"
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "${study}_capacity_receipts/exit_status.txt")" = 0
test ! -e "$receipt"; mkdir "$receipt"
finish(){ code=$?; echo "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
cfg=src/dunhuang/research/motion_bridge/config.json
"$edge" -m dunhuang.research.motion_bridge.study cache --config "$cfg" --capacity "$study/capacity" --out "$study/cache"
"$edge" -m dunhuang.research.motion_bridge.study cache --config "$cfg" --capacity "$study/capacity" --out "$study/cache_replay" --replay-from "$study/cache"
for arm in B3 B2; do
 "$edge" -m dunhuang.research.motion_bridge.train --config "$cfg" --capacity "$study/capacity" --cache "$study/cache" --out "$study/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.motion_bridge.train --config "$cfg" --capacity "$study/capacity" --cache "$study/cache" --out "$study/${arm}_replay" --arm "$arm" --replay-from "$study/$arm"
done
