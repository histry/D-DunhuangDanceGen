#!/usr/bin/env bash
set -euo pipefail
study="$1";receipt="${study}_receipts"
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test ! -e "$study";test ! -e "$receipt";mkdir -p "$study" "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
"$native" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$edge" -m pytest -q tests/test_root_pose_r15.py tests/test_boundary_state_r14.py > "$receipt/edge_tests.log" 2>&1
cfg=src/dunhuang/research/root_pose_r15/config.json
"$native" -m dunhuang.research.root_pose_r15.preflight --config "$cfg" --out "$study/preflight"
"$native" -m dunhuang.research.root_pose_r15.preflight --config "$cfg" --out "$study/preflight_replay" --replay-from "$study/preflight"
"$edge" -m dunhuang.research.root_pose_r15.probes --config "$cfg" --preflight "$study/preflight" --out "$study/probes"
"$edge" -m dunhuang.research.root_pose_r15.probes --config "$cfg" --preflight "$study/preflight" --out "$study/probes_replay" --replay-from "$study/probes"
"$native" -m dunhuang.research.root_pose_r15.evaluate --config "$cfg" --preflight "$study/preflight" --probes "$study/probes" --out "$study/native"
"$native" -m dunhuang.research.root_pose_r15.evaluate --config "$cfg" --preflight "$study/preflight" --probes "$study/probes" --out "$study/native_replay" --replay-from "$study/native"
