#!/usr/bin/env bash
set -euo pipefail
audit="$1"
study="$2"
attempt="${3:?fresh post attempt required}"
[[ "$attempt" =~ ^[0-9][0-9]$ ]]
receipt="$study"_post_"$attempt"_receipts
native=/home/disk/lsm/conda_envs/dance-rag/bin/python
edge=/home/disk/lsm/conda_envs/edge/bin/python
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
test "$(cat "$study"_receipts/exit_status.txt)" = 0
test ! -e "$receipt"; mkdir "$receipt"
finish(){ code=$?;echo "$code" > "$receipt/exit_status.txt";date -u +%FT%TZ > "$receipt/finished_utc.txt"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git rev-parse HEAD > "$receipt/execution_commit.txt"
date -u +%FT%TZ > "$receipt/started_utc.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$receipt/gpu.txt"
for environment in "$native" "$edge"; do
 label=$(basename "$(dirname "$(dirname "$environment")")")
 "$environment" -c 'import sys,json,torch,numpy;print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,cuda=torch.version.cuda,executable=sys.executable)))' > "$receipt/$label.json"
done
"$edge" -m dunhuang.research.condition_dependency.probe --training "$study" --out "$study/feature_probe_$attempt"
if test -f "$study/C2/result.json"; then
 "$native" -m dunhuang.research.condition_dependency.post --training "$study" --out "$study/media_$attempt" --comparison outputs/content_r10_01/generation
else
 "$native" -m dunhuang.research.condition_dependency.post --training "$study" --out "$study/media_$attempt"
fi
