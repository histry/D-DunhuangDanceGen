#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipts}"
test "$(cat ${root}_initial_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cfg=src/dunhuang/research/state_r27/config.json
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo positive_evaluation > "$receipt/phase.txt"
"$edge" -m dunhuang.research.state_r27.evaluate --config "$cfg" --root "$root" --out "$root/positive" --models R26_C1 P
"$edge" -m dunhuang.research.state_r27.evaluate --config "$cfg" --root "$root" --out "$root/positive_replay" --models R26_C1 P --replay-from "$root/positive"
"$native" -m dunhuang.research.state_r27.route --config "$cfg" --root "$root"
route=$("$native" -c 'import json,sys;print(json.load(open(sys.argv[1]))["route"])' "$root/decision.json")
if test "$route" = capacity;then
 echo nonlinear_and_capacity > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.state_r27.nonlinear --config "$cfg" --root "$root" --out "$root/nonlinear"
 "$edge" -m dunhuang.research.state_r27.nonlinear --config "$cfg" --root "$root" --out "$root/nonlinear_replay" --replay-from "$root/nonlinear"
 "$edge" -m dunhuang.research.state_r27.train --config "$cfg" --root "$root" --out "$root/models/P2" --arm P2
 "$edge" -m dunhuang.research.state_r27.train --config "$cfg" --root "$root" --out "$root/models/P2_replay" --arm P2 --replay-from "$root/models/P2"
 "$edge" -m dunhuang.research.state_r27.evaluate --config "$cfg" --root "$root" --out "$root/post_positive" --models P2
 "$edge" -m dunhuang.research.state_r27.evaluate --config "$cfg" --root "$root" --out "$root/post_positive_replay" --models P2 --replay-from "$root/post_positive"
elif test "$route" = actual_state;then
 echo actual_state_training > "$receipt/phase.txt"
 for arm in Q0 Q1;do
  "$edge" -m dunhuang.research.state_r27.train --config "$cfg" --root "$root" --out "$root/models/$arm" --arm "$arm"
  "$edge" -m dunhuang.research.state_r27.train --config "$cfg" --root "$root" --out "$root/models/${arm}_replay" --arm "$arm" --replay-from "$root/models/$arm"
 done
 "$edge" -m dunhuang.research.state_r27.cache_replay --config "$cfg" --root "$root" --out "$root/cache_replay"
 "$edge" -m dunhuang.research.state_r27.evaluate --config "$cfg" --root "$root" --out "$root/post_positive" --models Q0 Q1
 "$edge" -m dunhuang.research.state_r27.evaluate --config "$cfg" --root "$root" --out "$root/post_positive_replay" --models Q0 Q1 --replay-from "$root/post_positive"
fi
"$native" -m dunhuang.research.state_r27.branch_summary --root "$root"
echo completed > "$receipt/phase.txt"
