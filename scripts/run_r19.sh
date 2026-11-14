#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test ! -e "$study";test ! -e "$receipt";mkdir "$study" "$receipt" "$study/models"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
cfg=src/dunhuang/research/objective_r19/config.json
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_objective_r19.py > "$receipt/edge_tests.log" 2>&1
echo calibration > "$receipt/phase.txt"
"$edge" -m dunhuang.research.objective_r19.calibrate --config "$cfg" --out "$study/calibration"
"$edge" -m dunhuang.research.objective_r19.calibrate --config "$cfg" --out "$study/calibration_replay" --replay-from "$study/calibration"
echo training > "$receipt/phase.txt"
for arm in N0 N1 N2;do
 "$edge" -m dunhuang.research.objective_r19.train --config "$cfg" --calibration "$study/calibration" --out "$study/models/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.objective_r19.train --config "$cfg" --calibration "$study/calibration" --out "$study/models/${arm}_replay" --arm "$arm" --replay-from "$study/models/$arm"
done
"$native" -m dunhuang.research.objective_r19.gate --config "$cfg" --root "$study"
if "$native" -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["unique_admitted"])' "$study/gate.json";then
 "$edge" -m dunhuang.research.objective_r19.train --config "$cfg" --calibration "$study/calibration" --out "$study/models/U" --arm U
 "$edge" -m dunhuang.research.objective_r19.train --config "$cfg" --calibration "$study/calibration" --out "$study/models/U_replay" --arm U --replay-from "$study/models/U"
fi
echo rollout > "$receipt/phase.txt"
"$edge" -m dunhuang.research.objective_r19.rollout --config "$cfg" --root "$study" --out "$study/rollout"
"$edge" -m dunhuang.research.objective_r19.rollout --config "$cfg" --root "$study" --out "$study/rollout_replay" --replay-from "$study/rollout"
echo native > "$receipt/phase.txt"
"$native" -m dunhuang.research.objective_r19.evaluate --config "$cfg" --root "$study" --out "$study/native"
"$native" -m dunhuang.research.objective_r19.evaluate --config "$cfg" --root "$study" --out "$study/native_replay" --replay-from "$study/native"
"$native" scripts/verify_r19.py --root "$study"
echo videos > "$receipt/phase.txt"
"$native" scripts/render_r19.py --root "$study" --out "$study/videos"
echo completed > "$receipt/phase.txt"
