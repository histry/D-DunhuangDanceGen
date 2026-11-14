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
cfg=src/dunhuang/research/temporal_r20/config.json
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_temporal_r20.py > "$receipt/edge_tests.log" 2>&1
echo interface > "$receipt/phase.txt"
"$edge" -m dunhuang.research.temporal_r20.interface --config "$cfg" --out "$study/interface"
"$edge" -m dunhuang.research.temporal_r20.interface --config "$cfg" --out "$study/interface_replay" --replay-from "$study/interface"
for arm in S0 S1;do
 echo "structure_$arm" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.temporal_r20.train --config "$cfg" --root "$study" --out "$study/models/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.temporal_r20.train --config "$cfg" --root "$study" --out "$study/models/${arm}_replay" --arm "$arm" --replay-from "$study/models/$arm"
done
echo structure_sampling > "$receipt/phase.txt"
"$edge" -m dunhuang.research.temporal_r20.rollout --config "$cfg" --root "$study" --out "$study/structure_rollout" --part structure
"$edge" -m dunhuang.research.temporal_r20.rollout --config "$cfg" --root "$study" --out "$study/structure_rollout_replay" --part structure --replay-from "$study/structure_rollout"
"$native" -m dunhuang.research.temporal_r20.evaluate --config "$cfg" --root "$study" --out "$study/structure_native" --part structure
"$native" -m dunhuang.research.temporal_r20.evaluate --config "$cfg" --root "$study" --out "$study/structure_native_replay" --part structure --replay-from "$study/structure_native"
"$native" -m dunhuang.research.temporal_r20.gates --config "$cfg" --root "$study" --which structure
echo temporal_calibration > "$receipt/phase.txt"
"$edge" -m dunhuang.research.temporal_r20.calibrate --config "$cfg" --root "$study" --out "$study/temporal_calibration"
"$edge" -m dunhuang.research.temporal_r20.calibrate --config "$cfg" --root "$study" --out "$study/temporal_calibration_replay" --replay-from "$study/temporal_calibration"
for arm in A0 A1 A2;do
 echo "temporal_$arm" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.temporal_r20.train --config "$cfg" --root "$study" --out "$study/models/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.temporal_r20.train --config "$cfg" --root "$study" --out "$study/models/${arm}_replay" --arm "$arm" --replay-from "$study/models/$arm"
done
"$edge" -m dunhuang.research.temporal_r20.rollout --config "$cfg" --root "$study" --out "$study/gap_rollout" --part gap
"$edge" -m dunhuang.research.temporal_r20.rollout --config "$cfg" --root "$study" --out "$study/gap_rollout_replay" --part gap --replay-from "$study/gap_rollout"
"$native" -m dunhuang.research.temporal_r20.gates --config "$cfg" --root "$study" --which cache
if "$native" -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["cache_admitted"])' "$study/cache_gate.json";then
 echo temporal_A3 > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.temporal_r20.train --config "$cfg" --root "$study" --out "$study/models/A3" --arm A3
 "$edge" -m dunhuang.research.temporal_r20.train --config "$cfg" --root "$study" --out "$study/models/A3_replay" --arm A3 --replay-from "$study/models/A3"
fi
for part in p0 final;do
 echo "sampling_$part" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.temporal_r20.rollout --config "$cfg" --root "$study" --out "$study/${part}_rollout" --part "$part"
 "$edge" -m dunhuang.research.temporal_r20.rollout --config "$cfg" --root "$study" --out "$study/${part}_rollout_replay" --part "$part" --replay-from "$study/${part}_rollout"
 "$native" -m dunhuang.research.temporal_r20.evaluate --config "$cfg" --root "$study" --out "$study/${part}_native" --part "$part"
 "$native" -m dunhuang.research.temporal_r20.evaluate --config "$cfg" --root "$study" --out "$study/${part}_native_replay" --part "$part" --replay-from "$study/${part}_native"
done
"$native" scripts/verify_r20.py --root "$study"
echo videos > "$receipt/phase.txt"
"$native" scripts/render_r20.py --root "$study" --out "$study/videos"
echo completed > "$receipt/phase.txt"
