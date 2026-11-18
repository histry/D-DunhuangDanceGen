set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
export PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=outputs/residual_sdmr_r54_02/receipts
P=/home/disk/lsm/conda_envs/edge/bin/python
trap 's=$?; printf "exit_status=%s\ncommit=%s\nfinished=%s\n" "$s" "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" > "$R/closure_receipt.txt"' EXIT
$P -m pytest -q > "$R/pytest_final.log" 2>&1
$P - <<'CHECK'
from pathlib import Path
from dunhuang.research.complete_r36.core import read,write,sha
counts={}
for attempt in ('01','02'):
 root=Path('outputs/residual_sdmr_r54_'+attempt);files=read(root/'publication/seal.json')['files']
 for f in files:assert Path(f['path']).stat().st_size==f['bytes'] and sha(f['path'])==f['sha256']
 counts[attempt]=len(files)
write('outputs/residual_sdmr_r54_02/receipts/final_recheck.json',dict(attempts=counts,all_hashes_verified=True,optimizer_calls=0))
CHECK
$P -m dunhuang.research.warp_coupling_r53 recheck --out "$R/r53_preserved.json"
