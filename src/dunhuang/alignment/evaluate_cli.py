import argparse,json
from pathlib import Path
import torch
from .dataset import load
from .model import RetrievalModel
from .evaluate import retrieval
from dunhuang.io import write,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();rows,arrays,norm=load(Path(args.manifest))
    torch.set_num_threads(4)
    model=RetrievalModel().cuda().eval()
    payload=torch.load(args.checkpoint,map_location='cuda',weights_only=True)
    if payload['norm']!=norm:raise ValueError('Evaluation normalization differs')
    model.load_state_dict(payload['state'])
    result={}
    for role in ('val','test'):
        ids=[i for i,r in enumerate(rows) if r['role']==role]
        result[role]=retrieval(model,rows,arrays,ids,'cuda',task='local')
    write(args.out,dict(checkpoint_sha256=sha(args.checkpoint),local=result,qualification_modified=False,
        task='same-song exact physical start, alternative time intervals are negatives',
        pairing_scope='release-zero alignment conditional, not independently calibrated event semantic truth',
        heldout_status='historically exposed development regression'))
if __name__=='__main__':main()
