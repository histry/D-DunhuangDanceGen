#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";receipt="${2:?fresh receipt}"
test "$(cat ${root}_temporal_receipts/exit_status.txt)" = 0
test ! -e "$receipt";mkdir "$receipt"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
edge=/home/disk/lsm/conda_envs/edge/bin/python
cfg=src/dunhuang/research/continuity_r30/config.json
git rev-parse HEAD > "$receipt/execution_commit.txt";date -u +%FT%TZ > "$receipt/started_utc.txt"
trap 'code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt";exit "$code"' EXIT
echo heldout_reading > "$receipt/phase.txt"
"$edge" -m dunhuang.research.continuity_r30.evaluate --config "$cfg" --root "$root" --out "$root/reading_holdout" --arms R1 --roles holdout
"$edge" -m dunhuang.research.continuity_r30.evaluate --config "$cfg" --root "$root" --out "$root/reading_holdout_replay" --arms R1 --roles holdout --replay-from "$root/reading_holdout"
echo temporal_and_Root_delta > "$receipt/phase.txt"
arms=();if test -f "$root/models/R2/result.json";then arms=(R2);fi
"$edge" -m dunhuang.research.continuity_r30.evaluate --config "$cfg" --root "$root" --out "$root/temporal_evaluation" --arms "${arms[@]}" --root-diagnostic
"$edge" -m dunhuang.research.continuity_r30.evaluate --config "$cfg" --root "$root" --out "$root/temporal_evaluation_replay" --arms "${arms[@]}" --root-diagnostic --replay-from "$root/temporal_evaluation"
echo preclip_support > "$receipt/phase.txt"
"$edge" -m dunhuang.research.continuity_r30.support --config "$cfg" --root "$root" --out "$root/support"
"$edge" -m dunhuang.research.continuity_r30.support --config "$cfg" --root "$root" --out "$root/support_replay" --replay-from "$root/support"
echo actual_gate_probe > "$receipt/phase.txt"
"$edge" -m dunhuang.research.continuity_r30.probe --config "$cfg" --root "$root" --out "$root/probe"
"$edge" -m dunhuang.research.continuity_r30.probe --config "$cfg" --root "$root" --out "$root/probe_replay" --replay-from "$root/probe"
echo ChangE > "$receipt/phase.txt"
"$edge" -m dunhuang.research.continuity_r30.change --config "$cfg" --root "$root" --out "$root/change"
"$edge" -m dunhuang.research.continuity_r30.change --config "$cfg" --root "$root" --out "$root/change_replay" --replay-from "$root/change"
echo completed > "$receipt/phase.txt"
