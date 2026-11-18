set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=outputs/residual_sdmr_r54_02/receipts
P=/home/disk/lsm/conda_envs/edge/bin/python
trap 's=$?; printf "exit_status=%s\ncommit=%s\nfinished=%s\n" "$s" "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" > "$R/checks_receipt.txt"' EXIT
$P -m dunhuang.research.residual_sdmr_r54 checks --out outputs/residual_sdmr_r54_02/checks
$P -m dunhuang.research.residual_sdmr_r54 fit_view --out outputs/residual_sdmr_r54_02/fit_view
$P -c 'import json; g=json.load(open("outputs/residual_sdmr_r54_02/checks/gate.json")); print("MAIN_ADMISSION",g["admitted"],"NO_BUDGET_OR_THRESHOLD_CHANGES")'
