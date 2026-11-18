set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=outputs/residual_sdmr_r54_01/receipts
trap 's=$?; printf "exit_status=%s\ncommit=%s\nfinished=%s\n" "$s" "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" > "$R/preview_v2_receipt.txt"' EXIT
/home/disk/lsm/conda_envs/edge/bin/python -m dunhuang.research.residual_sdmr_r54 preview --out outputs/residual_sdmr_r54_01/preview_v2
cmp outputs/residual_sdmr_r54_01/preview/viewer/data.js outputs/residual_sdmr_r54_01/preview_v2/viewer/data.js
