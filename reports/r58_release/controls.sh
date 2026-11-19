#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
ROOT=outputs/execution_planning_r58_01
exec > >(tee "$ROOT/receipts/controls.log") 2>&1
trap 'status=$?; printf "exit_status=%s\nfinished_at=%s\n" "$status" "$(date -Is)" >> "$ROOT/receipts/controls_receipt.txt"' EXIT
printf 'commit=%s\nstarted_at=%s\ntmux=%s\n' "$(git rev-parse HEAD)" "$(date -Is)" "$TMUX" > "$ROOT/receipts/controls_receipt.txt"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/disk/lsm/conda_envs/edge/bin/python
set -x
"$PY" -c "import json; assert json.load(open('$ROOT/receipts/main_browser.json'))['passed']; assert json.load(open('$ROOT/independent_main/result.json'))['arrays_bytes_exact']"
"$PY" -m dunhuang.research.execution_planning_r58 controls --out "$ROOT/controls"
"$PY" -m dunhuang.research.execution_planning_r58 verify --source "$ROOT/controls" --out "$ROOT/independent_controls"
"$PY" -m dunhuang.research.execution_planning_r58 support --out "$ROOT/support"
"$PY" -m dunhuang.research.execution_planning_r58 support --out "$ROOT/independent_support/rebuild"
"$PY" -c "import json,hashlib;from pathlib import Path;a=Path('$ROOT/support/result.json');b=Path('$ROOT/independent_support/rebuild/result.json');assert a.read_bytes()==b.read_bytes();Path('$ROOT/independent_support/result.json').write_text(json.dumps(dict(result_exact=True,optimizer_calls=0,source_sha256=hashlib.sha256(a.read_bytes()).hexdigest()))+'\n')"
"$PY" -m dunhuang.research.execution_planning_r58 recovery --out "$ROOT/recovery"
"$PY" -m dunhuang.research.execution_planning_r58 verify_recovery --source "$ROOT/recovery" --out "$ROOT/independent_recovery"
"$PY" scripts/review_r58.py
"$PY" -m pytest -q > "$ROOT/receipts/pytest_final.log" 2>&1
