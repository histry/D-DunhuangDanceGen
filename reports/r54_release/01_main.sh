set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=outputs/residual_sdmr_r54_01/receipts
P=/home/disk/lsm/conda_envs/edge/bin/python
trap 's=$?; printf "exit_status=%s\ncommit=%s\nfinished=%s\n" "$s" "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" > "$R/main_receipt.txt"' EXIT
$P -m dunhuang.research.residual_sdmr_r54 checks --out outputs/residual_sdmr_r54_01/checks
if $P -c 'import json; assert json.load(open("outputs/residual_sdmr_r54_01/checks/gate.json"))["admitted"]'; then
 $P -m dunhuang.research.residual_sdmr_r54 train --arm Conditional --out outputs/residual_sdmr_r54_01/Conditional
 $P -m dunhuang.research.residual_sdmr_r54 main_view --out outputs/residual_sdmr_r54_01/main_view
fi
