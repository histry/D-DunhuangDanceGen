#!/bin/bash
# Run inside tmux. Receipt describes process completion, never quality admission.
set -eu
cd /home/disk/lsm/storage/DunhuangDanceGen
stage=${1:?pilot, replay, controls, or full}
case "$stage" in
  pilot) options=();;
  replay) options=(--replay);;
  controls) options=(--controls);;
  full) options=(--full);;
  *) exit 64;;
esac
folder=outputs/omni_judge_r71_01/receipts
mkdir -p "$folder"
test ! -e "$folder/${stage}_exit.json"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export R71_STAGE="$stage"
export R71_START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
py=/home/disk/lsm/conda_envs/dancejudge_r71/bin/python
set +e
"$py" -m dunhuang.research.omni_judge_r71.judge "${options[@]}" > "$folder/$stage.log" 2>&1
status=$?
R71_STATUS="$status" "$py" - <<'PY'
import os,json,subprocess,hashlib,platform
from pathlib import Path
from datetime import datetime,timezone
stage=os.environ['R71_STAGE'];p=Path('outputs/omni_judge_r71_01/receipts');log=p/(stage+'.log')
row=dict(stage=stage,started_utc=os.environ['R71_START'],ended_utc=datetime.now(timezone.utc).isoformat(),
    exit_status=int(os.environ['R71_STATUS']),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    python=platform.python_version(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),human_reviewed=False)
(p/(stage+'_exit.json')).write_text(json.dumps(row,indent=2))
print(json.dumps(row))
PY
exit "$status"
