"""Freeze the existing global encoders and fit only the local synchronization head."""
import argparse,json,random,hashlib
from pathlib import Path
import numpy as np
import torch
from .model import RetrievalModel
from .dataset import load
from .evaluate import synchronization,sync_logits,retrieval
from dunhuang.io import checked,write,sha

def encoder_digest(model):
    digest=hashlib.sha256()
    for branch in (model.music,model.motion):
        for name,value in sorted(branch.state_dict().items()):
            digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def fit(reference,manifest,out,steps):
    out.mkdir(parents=True,exist_ok=False)
    rows,arrays,norm=load(manifest)
    torch.set_num_threads(4);torch.manual_seed(20261111);random.seed(20261111)
    model=RetrievalModel().cuda().eval()
    payload=torch.load(checked(reference),map_location='cuda',weights_only=True)
    model.load_state_dict(payload['state'],strict=True)
    if payload['norm']!=norm:raise ValueError('Training normalization differs from frozen global model')
    for p in model.parameters():p.requires_grad_(False)
    for p in model.sync.parameters():p.requires_grad_(True)
    before=encoder_digest(model);groups={}
    for i,row in enumerate(rows):
        if row['role']=='train':groups.setdefault((row['domain'],row['group']),[]).append(i)
    val=[i for i,r in enumerate(rows) if r['role']=='val'];test=[i for i,r in enumerate(rows) if r['role']=='test']
    optimizer=torch.optim.AdamW(model.sync.parameters(),lr=.001)
    history=[];best=-1.
    for step in range(1,steps+1):
        domain=random.choice(('finedance','aistpp'))
        keys=random.sample([k for k in groups if k[0]==domain],min(12,sum(k[0]==domain for k in groups)))
        ids=[random.choice(groups[k]) for k in keys]
        args=[torch.as_tensor(arrays[k][ids],device='cuda') for k in ('music','motion','mm','dm')]
        logits=sync_logits(model,*args)
        loss=torch.nn.functional.cross_entropy(logits,torch.full((len(ids),),2,device='cuda',dtype=torch.long))
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%250==0:
            metrics=synchronization(model,rows,arrays,val,'cuda')
            score=float(np.mean([v['aligned_vs_shift_win'] for v in metrics.values()]))
            history.append(dict(step=step,loss=float(loss.detach()),validation=metrics))
            if score>best:
                best=score;torch.save(dict(state=model.state_dict(),norm=norm,encoder_sha256=before,
                    global_checkpoint_sha256=reference['sha256'],step=step),out/'sync.pt')
            write(out/'history.json',history);print('SYNC_HEAD',step,score,flush=True)
    model.load_state_dict(torch.load(out/'sync.pt',map_location='cuda',weights_only=True)['state'])
    after=encoder_digest(model)
    if after!=before:raise ValueError('Frozen encoders changed')
    validation=synchronization(model,rows,arrays,val,'cuda')
    global_val=retrieval(model,rows,arrays,val,'cuda')
    qualified=all(v['group_bootstrap95'][0]>.5 for v in validation.values()) and all(v['recall1']>v['chance'] for v in global_val.values())
    write(out/'result.json',dict(qualified=qualified,qualification_basis='validation_only',
        retrieval_checkpoint_sha256=reference['sha256'],sync_checkpoint_sha256=sha(out/'sync.pt'),
        encoder_sha256=after,encoders_frozen=before==after,validation=validation,
        test=synchronization(model,rows,arrays,test,'cuda'),preference_status='NO_EXPERT_PREFERENCES',
        heldout_status='previously exposed development regression'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=1000)
    a=ap.parse_args();path=Path(a.checkpoint).resolve();fit(dict(path=str(path),sha256=sha(path)),Path(a.manifest),Path(a.out).resolve(),a.steps)
