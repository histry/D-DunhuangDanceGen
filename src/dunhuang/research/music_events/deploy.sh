#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
root="$PWD/outputs/music_events_01"
envpath=/home/disk/lsm/conda_envs/music_events_01
test ! -e "$envpath"
/home/disk/lsm/conda_envs/dancejudge_r71/bin/python -m venv --system-site-packages "$envpath"
python="$envpath/bin/python"
git -C "$root/assets/beat_this" checkout --detach b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c
git -C "$root/assets/all-in-one" checkout --detach 18e78903c0365147a2c5d4e5e57ebf88cb7d800e
git clone --quiet https://github.com/CPJKU/madmom.git "$root/assets/madmom"
git -C "$root/assets/madmom" rev-parse HEAD > "$root/receipts/madmom_commit.txt"
"$python" -m pip install 'numpy==1.23.5' 'scipy==1.11.4' 'numba==0.59.1' 'librosa==0.10.2.post1' 'Cython==0.29.37' 'soundfile==0.12.1' 'einops==0.8.1' 'rotary-embedding-torch==0.8.6' 'soxr==0.5.0.post1' 'hydra-core==1.3.2' 'huggingface_hub==0.29.3'
"$python" -m pip install --no-deps 'natten==0.17.5+torch260cu124' -f https://whl.natten.org
"$python" -m pip install --no-build-isolation "$root/assets/madmom"
"$python" -m pip install 'demucs==4.0.1'
"$python" -m pip install --no-deps "$root/assets/beat_this" "$root/assets/all-in-one"
"$python" -m pip freeze > "$root/receipts/pip_freeze.txt"
PYTHONPATH=src "$python" - <<'PY'
import sys,torch,json,numpy,natten,librosa
from pathlib import Path
from dunhuang.research.music_events.compat import install
install()
import beat_this,allin1
receipt=dict(status='IMPORT_PASS',python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,numpy=numpy.__version__,librosa=librosa.__version__,natten=natten.__version__,cuda_available=torch.cuda.is_available(),torch_inherited_read_only=True,original_environments_modified=False)
Path('outputs/music_events_01/receipts/environment.json').write_text(json.dumps(receipt,indent=2));print(receipt)
PY
