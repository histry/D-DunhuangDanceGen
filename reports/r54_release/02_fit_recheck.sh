set -euo pipefail
cd /home/disk/lsm/storage/DunhuangDanceGen
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=outputs/residual_sdmr_r54_02/receipts
trap 's=$?; printf "exit_status=%s\ncommit=%s\nfinished=%s\n" "$s" "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" > "$R/fit_recheck_receipt.txt"' EXIT
/home/disk/lsm/conda_envs/edge/bin/python - <<'PY'
import torch,numpy as np
from dunhuang.research.residual_sdmr_r54.common import ROOT,read,write,sha,save_arrays
from dunhuang.research.residual_sdmr_r54.learn import Study
from dunhuang.research.coverage_r34.common import forbid_optimizers
from dunhuang.research.sdmr_event_rank.scorer import retrieval_metrics
forbid_optimizers();s=Study();g=read(ROOT/'checks/gate.json');p=ROOT/'checks/model.pt';assert sha(p)==read(ROOT/'checks/result.json')['model_sha256'];net=s.model().eval();net.load_state_dict(torch.load(p,map_location=s.device,weights_only=True)['state']);arrays={};hits=[]
for i in g['fit_task_indices']:
 t=s.meta['tasks'][i];scores=s.evaluate(net,t)['Conditional'];arrays[str(i)]=scores;hits.append(retrieval_metrics(scores,t['positive'])['recall_at1'])
accuracy=float(np.mean(hits));assert accuracy==g['real_train_fit_R1'];save_arrays(ROOT/'independent/fit_final_scores.npz',arrays);write(ROOT/'independent/fit_final.json',dict(fixed_train_tasks=len(hits),recomputed_R1=accuracy,original_R1=g['real_train_fit_R1'],model_sha256=sha(p),optimizer_calls=0,scope='TRAIN_FIT_ONLY_NOT_POPULATION_MODEL'))
print('EIGHT_TASK_FIT_R1_INDEPENDENT_EXACT',accuracy)
PY
