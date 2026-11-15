#!/usr/bin/env bash
set -euo pipefail
study="${1:?fresh study root}";receipt="${2:?fresh receipts}"
test ! -e "$study";test ! -e "$receipt";mkdir -p "$study" "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cfg=src/dunhuang/research/lodge_adapter_r25/config.json
native=/home/disk/lsm/conda_envs/dance-rag/bin/python;edge=/home/disk/lsm/conda_envs/edge/bin/python
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?; echo "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; exit "$code"' EXIT
echo regression > "$receipt/phase.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_lodge_adapter_r25.py > "$receipt/edge_tests.log" 2>&1
echo data > "$receipt/phase.txt"
"$edge" -m dunhuang.research.lodge_adapter_r25.data --config "$cfg" --out "$study/data"
"$edge" -m dunhuang.research.lodge_adapter_r25.data --config "$cfg" --out "$study/data_replay" --replay-from "$study/data"
echo sanity > "$receipt/phase.txt"
"$edge" -m dunhuang.research.lodge_adapter_r25.diagnose --config "$cfg" --root "$study" --out "$study/sanity"
"$edge" -m dunhuang.research.lodge_adapter_r25.diagnose --config "$cfg" --root "$study" --out "$study/sanity_replay" --replay-from "$study/sanity"
for arm in event phase;do
 echo "training_$arm" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.lodge_adapter_r25.train --config "$cfg" --root "$study" --out "$study/models/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.lodge_adapter_r25.train --config "$cfg" --root "$study" --out "$study/models/${arm}_replay" --arm "$arm" --replay-from "$study/models/$arm"
done
echo trained_graph > "$receipt/phase.txt"
"$edge" -m dunhuang.research.lodge_adapter_r25.diagnose --config "$cfg" --root "$study" --out "$study/trained_graph" --trained
"$edge" -m dunhuang.research.lodge_adapter_r25.diagnose --config "$cfg" --root "$study" --out "$study/trained_graph_replay" --trained --replay-from "$study/trained_graph"
echo sampling > "$receipt/phase.txt"
"$edge" -m dunhuang.research.lodge_adapter_r25.sample --config "$cfg" --root "$study" --out "$study/samples"
"$edge" -m dunhuang.research.lodge_adapter_r25.sample --config "$cfg" --root "$study" --out "$study/samples_replay" --replay-from "$study/samples"
echo evaluation > "$receipt/phase.txt"
"$native" -m dunhuang.research.lodge_adapter_r25.evaluate --config "$cfg" --root "$study" --out "$study/evaluation"
"$native" -m dunhuang.research.lodge_adapter_r25.evaluate --config "$cfg" --root "$study" --out "$study/evaluation_replay" --replay-from "$study/evaluation"
echo completed > "$receipt/phase.txt"
