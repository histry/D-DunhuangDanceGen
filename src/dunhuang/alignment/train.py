import copy
import random
import numpy as np
import torch
from torch.nn import functional as F
from .model import RetrievalModel,contrastive
from .evaluate import retrieval,synchronization,sync_logits
from dunhuang.io import write,sha

def tensors(arrays,ids,device):
    return [torch.as_tensor(arrays[k][ids],device=device) for k in ('music','motion','mm','dm')]

def fit(rows,arrays,norm,output,steps=5000,sync_steps=2000):
    torch.manual_seed(20261111);random.seed(20261111);np.random.seed(20261111)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda'
    model=RetrievalModel().to(device)
    groups={}
    for i,row in enumerate(rows):
        if row['role']=='train':groups.setdefault((row['domain'],row['group']),[]).append(i)
    domain_groups={d:[k for k in groups if k[0]==d] for d in ('finedance','aistpp')}
    train_ids=[i for i,r in enumerate(rows) if r['role']=='train']
    val=[i for i,r in enumerate(rows) if r['role']=='val']
    test=[i for i,r in enumerate(rows) if r['role']=='test']
    def batch():
        d=random.choice(list(domain_groups))
        selected=random.sample(domain_groups[d],min(24,len(domain_groups[d])))
        return [random.choice(groups[k]) for k in selected]
    history=[];best=-1;stale=0
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    for step in range(1,steps+1):
        ids=batch();music,motion,mm,dm=tensors(arrays,ids,device)
        model.train();m=model.music(music,mm)[0];d=model.motion(motion[:,15:135],dm[:,15:135])[0]
        loss=contrastive(m/model.log_temperature.exp().clamp(.02,.2),d,[rows[i]['group'] for i in ids])
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%250==0:
            model.eval();metrics=retrieval(model,rows,arrays,val,device)
            score=np.mean([v['mrr'] for v in metrics.values()])
            history.append(dict(stage='retrieval',step=step,loss=float(loss),val=metrics))
            write(output/'history.json',history)
            if score>best+1e-5:
                best=score;stale=0
                torch.save(dict(state=model.state_dict(),norm=norm,step=step),output/'retrieval.pt')
            else:stale+=1
            print('RETRIEVAL',step,float(loss),score,flush=True)
            if stale>=6:break
    payload=torch.load(output/'retrieval.pt',map_location=device,weights_only=True)
    model.load_state_dict(payload['state']);model.eval()
    global_test=retrieval(model,rows,arrays,test,device)
    global_val=retrieval(model,rows,arrays,val,device)
    sync=copy.deepcopy(model)
    for parameter in sync.parameters():parameter.requires_grad_(False)
    for parameter in sync.sync.parameters():parameter.requires_grad_(True)
    optimizer=torch.optim.AdamW(sync.sync.parameters(),lr=5e-5,weight_decay=.01)
    best=-1;stale=0
    for step in range(1,sync_steps+1):
        ids=batch();args=tensors(arrays,ids,device)
        sync.eval();logits=sync_logits(sync,*args)
        loss=F.cross_entropy(logits,torch.full((len(ids),),2,device=device,dtype=torch.long))
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(sync.parameters(),1.);optimizer.step()
        if step%250==0:
            sync.eval();metrics=synchronization(sync,rows,arrays,val,device)
            score=np.mean([v['aligned_vs_shift_win'] for v in metrics.values()])
            history.append(dict(stage='sync',step=step,loss=float(loss),val=metrics))
            write(output/'history.json',history)
            if score>best+1e-5:
                best=score;stale=0;torch.save(dict(state=sync.state_dict(),norm=norm,step=step,global_checkpoint_sha256=sha(output/'retrieval.pt')),output/'sync.pt')
            else:stale+=1
            print('SYNC',step,float(loss),score,flush=True)
            if stale>=4:break
    payload=torch.load(output/'sync.pt',map_location=device,weights_only=True)
    sync.load_state_dict(payload['state']);sync.eval()
    sync_test=synchronization(sync,rows,arrays,test,device)
    sync_val=synchronization(sync,rows,arrays,val,device)
    for branch in ('music','motion'):
        for name,value in getattr(model,branch).state_dict().items():
            if not torch.equal(value,getattr(sync,branch).state_dict()[name]):
                raise ValueError('Synchronization training changed frozen encoders')
    qualified=all(v['recall1']>v['chance'] for v in global_val.values()) and all(v['group_bootstrap95'][0]>.5 for v in sync_val.values())
    report=dict(qualification_basis='validation_only',frozen_sync_encoders=True,sync_val=sync_val,retrieval_val=global_val,retrieval_test=global_test,sync_test=sync_test,
        qualified=qualified,preference_status='NO_EXPERT_PREFERENCES',heldout_status='exposed development regression',
        retrieval_checkpoint_sha256=sha(output/'retrieval.pt'),sync_checkpoint_sha256=sha(output/'sync.pt'),
        ablation_status='pending_after_full_framework',train_steps=history[-1]['step'],normalization=norm)
    write(output/'result.json',report)
    return model,report
