#!/usr/bin/env bash
set -euo pipefail
study="${1:?fresh study}";receipt="${2:?fresh receipts}"
mode="${3:-new}"
test ! -e "$receipt"
if test "$mode" = new;then test ! -e "$study";mkdir "$study";else test "$mode" = resume_data;test -f "$study/recovery_inputs.json";test ! -e "$study/models";fi
mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cfg=src/dunhuang/research/reference_r26/config.json
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_reference_r26.py > "$receipt/edge_tests.log" 2>&1
if test "$mode" = new;then
echo data > "$receipt/phase.txt"
"$edge" -m dunhuang.research.reference_r26.data --config "$cfg" --out "$study/data"
"$edge" -m dunhuang.research.reference_r26.data --config "$cfg" --out "$study/data_replay" --replay-from "$study/data"
pf=preflight
else
"$native" -c 'import json,hashlib,sys;from pathlib import Path;r=Path(sys.argv[1]);d=json.loads((r/"recovery_inputs.json").read_text());assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in d["sealed_inputs"].items())' "$study"
pf=preflight_02
fi
echo preflight > "$receipt/phase.txt"
"$edge" -m dunhuang.research.reference_r26.preflight --config "$cfg" --root "$study" --out "$study/$pf"
"$edge" -m dunhuang.research.reference_r26.preflight --config "$cfg" --root "$study" --out "$study/${pf}_replay" --replay-from "$study/$pf"
for arm in C0 C1 C2;do
 echo "training_$arm" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.reference_r26.train --config "$cfg" --root "$study" --out "$study/models/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.reference_r26.train --config "$cfg" --root "$study" --out "$study/models/${arm}_replay" --arm "$arm" --replay-from "$study/models/$arm"
done
echo completed > "$receipt/phase.txt"
