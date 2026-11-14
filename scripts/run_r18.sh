#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test ! -e "$study";test ! -e "$receipt";mkdir "$study" "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
cfg=src/dunhuang/research/task_denoising_r18/config.json
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_task_denoising_r18.py > "$receipt/edge_tests.log" 2>&1
echo frozen_p0 > "$receipt/phase.txt"
"$edge" -m dunhuang.research.task_denoising_r18.audit --config "$cfg" --out "$study/p0"
"$edge" -m dunhuang.research.task_denoising_r18.audit --config "$cfg" --out "$study/p0_replay" --replay-from "$study/p0"
mkdir "$study/tiny"
echo tiny_train > "$receipt/phase.txt"
for mode in self unique;do
 "$edge" -m dunhuang.research.task_denoising_r18.train --config "$cfg" --out "$study/tiny/$mode" --mode "$mode" --stage tiny
 "$edge" -m dunhuang.research.task_denoising_r18.train --config "$cfg" --out "$study/tiny/${mode}_replay" --mode "$mode" --stage tiny --replay-from "$study/tiny/$mode"
done
"$native" -m dunhuang.research.task_denoising_r18.gate --config "$cfg" --root "$study"
echo tiny_rollout > "$receipt/phase.txt"
"$edge" -m dunhuang.research.task_denoising_r18.rollout --config "$cfg" --root "$study" --out "$study/tiny_rollout" --stage tiny
"$edge" -m dunhuang.research.task_denoising_r18.rollout --config "$cfg" --root "$study" --out "$study/tiny_rollout_replay" --stage tiny --replay-from "$study/tiny_rollout"
if "$native" -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["main_admitted"])' "$study/tiny_gate.json";then
 mkdir "$study/main"
 echo main_train > "$receipt/phase.txt"
 for mode in self unique;do
  "$edge" -m dunhuang.research.task_denoising_r18.train --config "$cfg" --out "$study/main/$mode" --mode "$mode" --stage main
  "$edge" -m dunhuang.research.task_denoising_r18.train --config "$cfg" --out "$study/main/${mode}_replay" --mode "$mode" --stage main --replay-from "$study/main/$mode"
 done
 "$edge" -m dunhuang.research.task_denoising_r18.rollout --config "$cfg" --root "$study" --out "$study/main_rollout" --stage main
 "$edge" -m dunhuang.research.task_denoising_r18.rollout --config "$cfg" --root "$study" --out "$study/main_rollout_replay" --stage main --replay-from "$study/main_rollout"
else
 echo "Main not admitted by fixed Train-only gate; no extra tuning" > "$receipt/main_skipped.txt"
fi
echo native_replay > "$receipt/phase.txt"
"$native" -m dunhuang.research.task_denoising_r18.evaluate --config "$cfg" --root "$study" --out "$study/native"
"$native" -m dunhuang.research.task_denoising_r18.evaluate --config "$cfg" --root "$study" --out "$study/native_replay" --replay-from "$study/native"
"$native" scripts/verify_r18.py --root "$study"
echo videos > "$receipt/phase.txt"
"$native" scripts/render_r18.py --root "$study" --out "$study/videos"
echo completed > "$receipt/phase.txt"
