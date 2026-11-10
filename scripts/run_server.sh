#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
export PYTHONPATH="$PWD/src"
/home/disk/lsm/conda_envs/edge/bin/python -m pytest -q tests > outputs/tests.log 2>&1
set +e
/home/disk/lsm/conda_envs/dance-rag/bin/python scripts/run_experiments.py > outputs/experiment.log 2>&1
status=$?
printf '%s\n' "$status" > outputs/experiment.exit
exit "$status"
