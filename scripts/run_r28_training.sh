#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipts}"
initial_receipt="${3:-${root}_initial_receipts}"
test "$(cat $initial_receipt/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/contract_r28/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo FineDance_audio > "$receipt/phase.txt"
"$edge" -m dunhuang.research.contract_r28.audio_audit --config "$cfg" --root "$root" --out "$root/fine_audio"
"$edge" -m dunhuang.research.contract_r28.audio_audit --config "$cfg" --root "$root" --out "$root/fine_audio_replay" --replay-from "$root/fine_audio"
echo input_domain_audit > "$receipt/phase.txt"
"$edge" -m dunhuang.research.contract_r28.audit --config "$cfg" --root "$root" --out "$root/input_domain_audit.json"
"$edge" -m dunhuang.research.contract_r28.audit --config "$cfg" --root "$root" --out "$root/input_domain_audit_replay.json" --replay-from "$root/input_domain_audit.json"
"$native" -m dunhuang.research.contract_r28.admit --root "$root"
echo training_tests > "$receipt/phase.txt"
"$edge" -m pytest -q tests/test_contract_r28.py > "$receipt/edge_tests.log" 2>&1
for arm in T0 T1;do
 echo "training_$arm" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.contract_r28.train --config "$cfg" --root "$root" --out "$root/models/$arm" --arm "$arm"
 "$edge" -m dunhuang.research.contract_r28.train --config "$cfg" --root "$root" --out "$root/models/${arm}_replay" --arm "$arm" --replay-from "$root/models/$arm"
done
echo real_sampling > "$receipt/phase.txt"
"$edge" -m dunhuang.research.contract_r28.evaluate --config "$cfg" --root "$root" --out "$root/evaluation"
"$edge" -m dunhuang.research.contract_r28.evaluate --config "$cfg" --root "$root" --out "$root/evaluation_replay" --replay-from "$root/evaluation"
echo completed > "$receipt/phase.txt"
