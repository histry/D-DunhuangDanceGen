#!/bin/bash
cd /home/disk/lsm/storage/DunhuangDanceGen || exit 90
set -e
folder=outputs/omni_judge_r71_01
mkdir -p "$folder/receipts" "$folder/assets"
exec > "$folder/receipts/setup.log" 2>&1
envpath=/home/disk/lsm/conda_envs/dancejudge_r71
if [ ! -x "$envpath/bin/python" ]; then /home/disk/lsm/miniconda3/bin/conda create -p "$envpath" python=3.10 -y; fi
"$envpath/bin/python" -m pip install 'torch==2.6.0' 'torchvision==0.21.0' 'transformers==4.52.3' 'autoawq==0.2.9' accelerate 'qwen-omni-utils[decord]' soundfile matplotlib scipy pillow
git clone https://github.com/QwenLM/Qwen2.5-Omni.git "$folder/assets/Qwen2.5-Omni"
git -C "$folder/assets/Qwen2.5-Omni" rev-parse HEAD > "$folder/receipts/official_commit.txt"
"$envpath/bin/python" - <<'PY'
from huggingface_hub import HfApi,snapshot_download
import json,hashlib
from pathlib import Path
p=Path('outputs/omni_judge_r71_01');info=HfApi().model_info('Qwen/Qwen2.5-Omni-7B-AWQ',files_metadata=True)
(p/'receipts/model_registration.json').write_text(json.dumps(dict(model_id=info.id,revision=info.sha,files=[dict(path=f.rfilename,size=f.size,lfs_sha256=f.lfs.sha256 if f.lfs else None) for f in info.siblings]),indent=2))
path=snapshot_download(info.id,revision=info.sha,local_dir=p/'assets/model')
files=[dict(path=f.relative_to(Path(path)).as_posix(),size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in Path(path).rglob('*') if f.is_file() and '.cache' not in f.parts]
(p/'receipts/model_files.json').write_text(json.dumps(files,indent=2))
print('R71_MODEL_READY',flush=True)
PY
"$envpath/bin/python" -m pip freeze > "$folder/receipts/pip_freeze.txt"
