set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
R=outputs/residual_sdmr_r54_02/receipts
trap 's=$?; printf "exit_status=%s\ncommit=%s\nfinished=%s\n" "$s" "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" > "$R/public_export_receipt.txt"' EXIT
/home/disk/lsm/conda_envs/edge/bin/python scripts/release_r54.py --out outputs/residual_sdmr_r54_02/public_export
