#!/usr/bin/env bash
set -euo pipefail
study="${1:?fresh study root}";receipt="${2:?fresh receipt root}"
cfg=src/dunhuang/research/local_r23/config.json
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
test ! -e "$study";test ! -e "$receipt";mkdir -p "$study" "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?; echo "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_local_r23.py > "$receipt/edge_tests.log" 2>&1
for stage in gradients sample;do
 echo "$stage" > "$receipt/phase.txt"
 folder="$stage";test "$stage" != sample || folder=samples
 "$edge" -m dunhuang.research.local_r23.$stage --config "$cfg" --out "$study/$folder"
 "$edge" -m dunhuang.research.local_r23.$stage --config "$cfg" --out "$study/${folder}_replay" --replay-from "$study/$folder"
done
for stage in represent external physical_tracking;do
 echo "$stage" > "$receipt/phase.txt"
 py="$native";test "$stage" != external || py="$edge"
 "$py" -m dunhuang.research.local_r23.$stage --config "$cfg" --out "$study/$stage"
 "$py" -m dunhuang.research.local_r23.$stage --config "$cfg" --out "$study/${stage}_replay" --replay-from "$study/$stage"
done
echo native > "$receipt/phase.txt"
"$native" -m dunhuang.research.local_r23.evaluate --config "$cfg" --study "$study" --out "$study/native"
"$native" -m dunhuang.research.local_r23.evaluate --config "$cfg" --study "$study" --out "$study/native_replay" --replay-from "$study/native"
echo independent_checks > "$receipt/phase.txt"
"$native" scripts/verify_r23.py --root "$study"
echo completed > "$receipt/phase.txt"
