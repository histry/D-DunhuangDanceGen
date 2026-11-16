#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test "$(cat ${root}_reading_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/continuity_r30/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo Train_only_calibration > "$receipt/phase.txt"
"$edge" -m dunhuang.research.continuity_r30.calibrate --config "$cfg" --root "$root" --out "$root/calibration"
"$edge" -m dunhuang.research.continuity_r30.calibrate --config "$cfg" --root "$root" --out "$root/calibration_replay" --replay-from "$root/calibration"
echo temporal_training > "$receipt/phase.txt"
if "$edge" -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))["temporal_training_admitted"] else 1)' "$root/calibration/result.json";then
"$edge" -m dunhuang.research.continuity_r30.train --config "$cfg" --root "$root" --out "$root/models/R2" --arm R2
"$edge" -m dunhuang.research.continuity_r30.train --config "$cfg" --root "$root" --out "$root/models/R2_replay" --arm R2 --replay-from "$root/models/R2"
else
 echo R30_TEMPORAL_NOT_ADMITTED
fi
echo completed > "$receipt/phase.txt"
