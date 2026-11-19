#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/paired_benchmark_r59_01
test -d "$ROOT"; test ! -e "$ROOT/data"; test ! -e "$ROOT/main"
mkdir -p "$ROOT/receipts"
cp /tmp/dunhuang_r59_main_retry.sh "$ROOT/receipts/main.sh"
exec > >(tee "$ROOT/receipts/main.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/main_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/main_receipt.txt"
test "$(git rev-parse HEAD)" = 15fdbea3cf77f3d9fab78f766741f9e57eddbe0d
export PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/disk/lsm/conda_envs/edge/bin/python
for i in $(seq 1 120); do
 if test -f inputs/paired_benchmark_r59/finedance/upload_receipt.json; then break; fi
 sleep 5
done
test -f inputs/paired_benchmark_r59/finedance/upload_receipt.json
set -x
"$PY" -m pytest tests/test_paired_benchmark_r59.py -q > "$ROOT/receipts/pytest_targeted.log" 2>&1
"$PY" -m dunhuang.research.paired_benchmark_r59 data --out "$ROOT/data"
"$PY" -m dunhuang.research.paired_benchmark_r59 main --out "$ROOT/main"
"$PY" -m dunhuang.research.paired_benchmark_r59 render --source "$ROOT/main" --out "$ROOT/main_video"
"$PY" -m dunhuang.research.paired_benchmark_r59 verify --source "$ROOT/main" --out "$ROOT/independent_main"
