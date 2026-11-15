#!/usr/bin/env bash
set -euo pipefail
study="$1"; receipt="${study}_receipts"
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
cfg=src/dunhuang/research/context_r21/config.json
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_context_r21.py tests/test_context_r21_publication.py > "$receipt/edge_tests.log" 2>&1
echo data > "$receipt/phase.txt"
"$native" -m dunhuang.research.context_r21.data --config "$cfg" --out "$study/data"
"$native" -m dunhuang.research.context_r21.data --config "$cfg" --out "$study/data_replay" --replay-from "$study/data"
echo neutrality > "$receipt/phase.txt"
"$edge" -m dunhuang.research.context_r21.neutrality --config "$cfg" --root "$study" --out "$study/neutrality"
"$edge" -m dunhuang.research.context_r21.neutrality --config "$cfg" --root "$study" --out "$study/neutrality_replay" --replay-from "$study/neutrality"
for arm in B0 B1 B2 B3;do
 echo "training_$arm" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.context_r21.train --config "$cfg" --root "$study" --out "$study/models/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.context_r21.train --config "$cfg" --root "$study" --out "$study/models/${arm}_replay" --arm "$arm" --replay-from "$study/models/$arm"
done
if "$native" -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["support_admitted"])' "$study/data/result.json";then
 echo support_calibration > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.context_r21.calibrate --config "$cfg" --root "$study" --out "$study/support_calibration"
 "$edge" -m dunhuang.research.context_r21.calibrate --config "$cfg" --root "$study" --out "$study/support_calibration_replay" --replay-from "$study/support_calibration"
fi
for arm in Q M F;do
 if test "$arm" = F && ! "$native" -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["support_admitted"])' "$study/data/result.json";then continue;fi
 echo "training_$arm" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.context_r21.train --config "$cfg" --root "$study" --out "$study/models/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.context_r21.train --config "$cfg" --root "$study" --out "$study/models/${arm}_replay" --arm "$arm" --replay-from "$study/models/$arm"
done
echo final_sampling > "$receipt/phase.txt"
"$edge" -m dunhuang.research.context_r21.rollout --config "$cfg" --root "$study" --out "$study/rollout"
"$edge" -m dunhuang.research.context_r21.rollout --config "$cfg" --root "$study" --out "$study/rollout_replay" --replay-from "$study/rollout"
echo native > "$receipt/phase.txt"
"$native" -m dunhuang.research.context_r21.evaluate --config "$cfg" --root "$study" --out "$study/native"
"$native" -m dunhuang.research.context_r21.evaluate --config "$cfg" --root "$study" --out "$study/native_replay" --replay-from "$study/native"
echo independent_checks > "$receipt/phase.txt"
"$native" scripts/verify_r21.py --root "$study"
echo videos > "$receipt/phase.txt"
"$native" scripts/render_r21.py --root "$study" --out "$study/videos"
echo completed > "$receipt/phase.txt"
