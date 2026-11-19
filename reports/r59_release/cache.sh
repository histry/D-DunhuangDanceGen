#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/paired_benchmark_r59_01
cp /tmp/dunhuang_r59_cache.sh "$ROOT/receipts/cache.sh"
exec > >(tee "$ROOT/receipts/cache.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/cache_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/cache_receipt.txt"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/disk/lsm/conda_envs/edge/bin/python -m dunhuang.research.paired_benchmark_r59 verify --source "$ROOT/main" --out "$ROOT/cache_qualification"
