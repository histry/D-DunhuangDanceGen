#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/paired_benchmark_r59_01
cp /tmp/dunhuang_r59_controls.sh "$ROOT/receipts/controls.sh"
exec > >(tee "$ROOT/receipts/controls.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/controls_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/controls_receipt.txt"
test "$(git rev-parse HEAD)" = 15fdbea3cf77f3d9fab78f766741f9e57eddbe0d
export PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/disk/lsm/conda_envs/edge/bin/python
set -x
"$PY" -c "import json;assert json.load(open('$ROOT/receipts/main_browser.json'))['passed'];assert json.load(open('$ROOT/independent_main/result.json'))['score_bytes_exact']"
"$PY" -m dunhuang.research.paired_benchmark_r59 verify_data --source "$ROOT/data" --out "$ROOT/independent_data"
"$PY" -m dunhuang.research.paired_benchmark_r59 controls --out "$ROOT/controls"
"$PY" -m dunhuang.research.paired_benchmark_r59 verify --source "$ROOT/controls" --out "$ROOT/independent_controls"
"$PY" -m dunhuang.research.paired_benchmark_r59 diagnostics --out "$ROOT/r58_diagnostics"
"$PY" -m dunhuang.research.paired_benchmark_r59 diagnostics --out "$ROOT/independent_diagnostics/rebuild"
"$PY" -c "import json;from pathlib import Path;a=json.load(open('$ROOT/r58_diagnostics/result.json'));b=json.load(open('$ROOT/independent_diagnostics/rebuild/result.json'));assert a==b;Path('$ROOT/independent_diagnostics/result.json').write_text(json.dumps(dict(result_exact=True,optimizer_calls=0))+'\n')"
"$PY" -m pytest -q > "$ROOT/receipts/pytest_final.log" 2>&1
