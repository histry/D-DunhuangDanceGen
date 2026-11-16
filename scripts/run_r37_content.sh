#!/usr/bin/env bash
set -euo pipefail
root="${1:?root}";first_media="${2:-$root/viewer_first}";test -f "$first_media/protocol.json";r="$root/receipts/content";test ! -e "$r";mkdir -p "$r"
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cpu=/home/disk/lsm/conda_envs/dance-rag/bin/python;gpu=/home/disk/lsm/conda_envs/edge/bin/python
sha256sum "$first_media/protocol.json" > "$r/first_media_protocol.txt";git rev-parse HEAD > "$r/code_commit.txt";date -u +%FT%TZ > "$r/started_utc.txt"
trap 'code=$?;echo "$code" > "$r/exit_status.txt";date -u +%FT%TZ > "$r/finished_utc.txt";exit "$code"' EXIT
echo verified_variable_pair_population > "$r/phase.txt"
"$cpu" -m dunhuang.research.transition_r37.content_prepare --out "$root/content_data" > "$r/data.log" 2>&1
echo one_fixed_main_training > "$r/phase.txt"
"$gpu" -m dunhuang.research.transition_r37.content_train --data "$root/content_data" --out "$root/content_model" > "$r/train.log" 2>&1
echo fresh_weight_evaluation_replay > "$r/phase.txt"
"$gpu" -m dunhuang.research.transition_r37.content_train --data "$root/content_data" --out "$root/content_replay" --replay-from "$root/content_model" > "$r/replay.log" 2>&1
echo completed > "$r/phase.txt"
