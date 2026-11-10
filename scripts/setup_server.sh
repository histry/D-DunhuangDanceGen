#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
legacy=/home/disk/lsm/storage/danceRAG
commit=465f35ba88bbc45d3415e623a60c55a2c8fdb49f
if [ ! -d third_party/danceRAG/.git ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --shared --no-checkout "$legacy" third_party/danceRAG
  GIT_LFS_SKIP_SMUDGE=1 git -C third_party/danceRAG checkout --detach "$commit"
fi
test "$(git -C third_party/danceRAG rev-parse HEAD)" = "$commit"
git -C third_party/danceRAG diff --exit-code HEAD -- >/dev/null
printf 'Pinned Native dependency ready. Existing data remains in the asset store.\n'
