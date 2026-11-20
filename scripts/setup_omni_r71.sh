#!/bin/bash
cd /home/disk/lsm/storage/DunhuangDanceGen || exit 90
set -e
folder=outputs/omni_judge_r71_01
mkdir -p "$folder/receipts" "$folder/assets"
exec > "$folder/receipts/${R71_SETUP_LOG:-setup.log}" 2>&1
envpath=/home/disk/lsm/conda_envs/dancejudge_r71
if [ ! -x "$envpath/bin/python" ]; then /home/disk/lsm/miniconda3/bin/conda create -p "$envpath" python=3.10 -y; fi
"$envpath/bin/python" -m pip install --find-links "$folder/assets/wheels" 'torch==2.6.0' 'torchvision==0.21.0' 'transformers==4.52.3' 'autoawq==0.2.9' accelerate qwen-omni-utils soundfile matplotlib scipy pillow
if [ ! -d "$folder/assets/Qwen2.5-Omni/.git" ]; then git clone https://github.com/QwenLM/Qwen2.5-Omni.git "$folder/assets/Qwen2.5-Omni"; fi
code_sha=$(git -C "$folder/assets/Qwen2.5-Omni" rev-parse HEAD)
if [ -f "$folder/receipts/official_commit.txt" ]; then test "$(cat "$folder/receipts/official_commit.txt")" = "$code_sha"; else echo "$code_sha" > "$folder/receipts/official_commit.txt"; fi
PYTHONPATH=src "$envpath/bin/python" -m dunhuang.research.omni_judge_r71.download_model
"$envpath/bin/python" -m pip uninstall -y decord
"$envpath/bin/python" -m pip check
"$envpath/bin/python" -m pip freeze > "$folder/receipts/pip_freeze.txt"
echo R71_ISOLATED_SETUP_READY