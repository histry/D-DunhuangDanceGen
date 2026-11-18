set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=outputs/residual_sdmr_r54_02/receipts
P=/home/disk/lsm/conda_envs/edge/bin/python
trap 's=$?; printf "exit_status=%s\ncommit=%s\nfinished=%s\n" "$s" "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" > "$R/prepare_receipt.txt"' EXIT
CUDA_VISIBLE_DEVICES='' $P -m pytest -q > "$R/pytest.log" 2>&1
$P -m dunhuang.research.residual_sdmr_r54 prepare --out outputs/residual_sdmr_r54_02/data
$P -m dunhuang.research.residual_sdmr_r54 preview --out outputs/residual_sdmr_r54_02/preview
