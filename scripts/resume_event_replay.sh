#!/usr/bin/env bash
set -euo pipefail
study="$1"
replay="$2"
receipt="$3"
source="$4"
python_bin=/home/disk/lsm/conda_envs/dance-rag/bin/python
mkdir -p "$receipt"
success=0
finish() { code=$?; if [ "$success" -ne 1 ] && [ "$code" -eq 0 ]; then code=125; fi; printf '%s\n' "$code" > "$receipt/exit_status.txt"; date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%FT%TZ > "$receipt/started_utc.txt"
git rev-parse HEAD > "$receipt/replay_commit.txt"
"$python_bin" -m pytest -q -rs > "$receipt/pytest.log" 2>&1
"$python_bin" scripts/verify_event_identifiability.py --source "$source" --out "$study" --replay-out "$replay"
"$python_bin" scripts/export_event_identifiability.py --out "$study" --receipt "$receipt" --dest "$study/public_summary.json"
"$python_bin" scripts/audit_event_identifiability.py --out "$study" --source "$source" --dest "$study/additional_audit" > "$receipt/audit.log"
success=1
