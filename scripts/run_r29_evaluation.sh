#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipts}"
test "$(cat ${root}_training_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/time_r29/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
for domain in aist fine;do
 echo "sampling_$domain" > "$receipt/phase.txt"
 "$edge" -m dunhuang.research.time_r29.evaluate --config "$cfg" --root "$root" --out "$root/evaluation_$domain" --domain "$domain"
 "$edge" -m dunhuang.research.time_r29.evaluate --config "$cfg" --root "$root" --out "$root/evaluation_${domain}_replay" --domain "$domain" --replay-from "$root/evaluation_$domain"
done
echo actual_state_probes > "$receipt/phase.txt"
"$edge" -m dunhuang.research.time_r29.probe --config "$cfg" --root "$root" --out "$root/probe"
"$edge" -m dunhuang.research.time_r29.probe --config "$cfg" --root "$root" --out "$root/probe_replay" --replay-from "$root/probe"
"$native" -m dunhuang.research.time_r29.decision --root "$root"
echo ChangE > "$receipt/phase.txt"
"$edge" -m dunhuang.research.time_r29.change --config "$cfg" --root "$root" --out "$root/change"
"$edge" -m dunhuang.research.time_r29.change --config "$cfg" --root "$root" --out "$root/change_replay" --replay-from "$root/change"
echo completed > "$receipt/phase.txt"
