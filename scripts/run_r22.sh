#!/usr/bin/env bash
set -euo pipefail
study="${1:?fresh output root required}"
receipt="${2:?fresh receipt root required}"
cfg=src/dunhuang/research/branch_r22/config.json
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
test ! -e "$study";test ! -e "$receipt";mkdir -p "$study" "$receipt"
export PYTHONPATH=src
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?; echo "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_branch_r22.py tests/test_branch_r22_publication.py > "$receipt/edge_tests.log" 2>&1
echo data > "$receipt/phase.txt"
"$native" -m dunhuang.research.branch_r22.data --config "$cfg" --root "$study" --out "$study/data"
"$native" -m dunhuang.research.branch_r22.data --config "$cfg" --root "$study" --out "$study/data_replay" --replay-from "$study/data"
echo sanity > "$receipt/phase.txt"
"$edge" -m dunhuang.research.branch_r22.sanity --config "$cfg" --root "$study" --out "$study/sanity"
"$edge" -m dunhuang.research.branch_r22.sanity --config "$cfg" --root "$study" --out "$study/sanity_replay" --replay-from "$study/sanity"
echo lodge > "$receipt/phase.txt"
"$edge" -m dunhuang.research.branch_r22.lodge --config "$cfg" --root "$study" --out "$study/lodge"
"$edge" -m dunhuang.research.branch_r22.lodge --config "$cfg" --root "$study" --out "$study/lodge_replay" --replay-from "$study/lodge"
for arm in G0 G1 G2;do
 echo "training_$arm" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.branch_r22.train --config "$cfg" --root "$study" --out "$study/models/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.branch_r22.train --config "$cfg" --root "$study" --out "$study/models/${arm}_replay" --arm "$arm" --replay-from "$study/models/$arm"
done
echo final_sampling > "$receipt/phase.txt"
"$edge" -m dunhuang.research.branch_r22.rollout --config "$cfg" --root "$study" --out "$study/rollout"
"$edge" -m dunhuang.research.branch_r22.rollout --config "$cfg" --root "$study" --out "$study/rollout_replay" --replay-from "$study/rollout"
echo native > "$receipt/phase.txt"
"$native" -m dunhuang.research.context_r21.evaluate --config "$cfg" --root "$study" --out "$study/native"
"$native" -m dunhuang.research.context_r21.evaluate --config "$cfg" --root "$study" --out "$study/native_replay" --replay-from "$study/native"
echo independent_checks > "$receipt/phase.txt"
"$native" scripts/verify_r22.py --root "$study"
echo videos > "$receipt/phase.txt"
"$native" scripts/render_r22.py --root "$study" --out "$study/videos"
echo completed > "$receipt/phase.txt"
