"""Fixed Train-only P2: identical initialization, data, updates; only rank labels differ."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.paired_event_r41.train import objective,seeded,digest,save
from dunhuang.research.paired_event_r41.population import Population
from dunhuang.research.coverage_r34.common import forbid_optimizers
from .features import load_model,inputs

ARMS=('NoRank','Rank','ShuffledRank')
WEIGHTS=dict(global_=1.,clock=.1,local=.1,reconstruction=.1,equivariance=.05)
WEIGHTS['global']=WEIGHTS.pop('global_')

def schedule(meta):
    rng=np.random.default_rng(meta['config']['seed']);tasks=meta['tasks'];order=[]
    for step in range(240):
        length=meta['config']['lengths'][step%3];groups={}
        for i,t in enumerate(tasks):
            if t['role']=='train' and t['pool']=='energy_matched' and t['frames']==length:groups.setdefault(t['source_group'],[]).append(i)
        if len(groups)<2:raise ValueError('Real independent Train groups required at every length')
        keys=sorted(groups);chosen=rng.choice(keys,min(4,len(keys)),replace=False);order.append([int(rng.choice(groups[k])) for k in chosen])
    return order

def shuffled_labels(positive,seed):
    rng=np.random.default_rng(seed);p=np.asarray(positive,bool);return p[rng.permutation(len(p))]

def rank_loss(model,pop,meta,task_ids,arm,step):
    tasks=[meta['tasks'][i] for i in task_ids];rows=meta['rows'];qids=[t['query'] for t in tasks];cids=[i for t in tasks for i in t['candidates']]
    mt,_,mv,_=model.content(*inputs(pop,rows,qids));_,dt,_,dv=model.content(*inputs(pop,rows,cids));parts=[];offset=0
    for j,t in enumerate(tasks):
        n=len(t['candidates']);valid=mv[j,::3][None]&dv[offset:offset+n,::3];sim=(mt[j,::3][None]*dt[offset:offset+n,::3]).sum(-1);scores=(sim*valid).sum(-1)/valid.sum(-1).clamp_min(1)
        if not valid.any(-1).all():raise ValueError('No valid real diagonal support')
        labels=shuffled_labels(t['positive'],meta['config']['seed']+step*10+j) if arm=='ShuffledRank' else t['positive'];positive=torch.tensor(labels,device='cuda');logits=scores/.1
        parts.append(torch.logsumexp(logits,0)-torch.logsumexp(logits[positive],0));offset+=n
    return torch.stack(parts).mean()

def loss(model,pop,meta,task_ids,arm,step):
    seeded(meta['config']['seed']+step);parents=[meta['rows'][meta['tasks'][i]['query']]['parent_index'] for i in task_ids]
    value,parts,detail=objective(model,pop,parents,dict(seed=meta['config']['seed'],loss_weights=WEIGHTS),step)
    rank=rank_loss(model,pop,meta,task_ids,arm,step);parts['rank']=rank;value=value+rank*(0. if arm=='NoRank' else 1.)
    return value,parts,detail

def gradients(parts,model):
    params=tuple(model.music.parameters())+tuple(model.motion.parameters());vectors={}
    for name,value in parts.items():
        g=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True);vectors[name]=torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(params,g)]).detach().cpu().numpy()
    return vectors

def run(benchmark,evaluation,arm,out,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh P2 directory')
    out.mkdir(parents=True);meta=read(Path(benchmark)/'manifest.json');gate=read(Path(evaluation)/'result.json')['p2_gate']
    if not gate['admitted'] or not gate['train_only']:raise ValueError('Frozen Train gate required')
    pop=Population(meta['config']['data']);order=schedule(meta);model=load_model(meta['config']);model.requires_grad_(True);model.train();initial=digest(model)
    protocol=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),benchmark_sha256=sha(Path(benchmark)/'manifest.json'),gate_sha256=sha(Path(evaluation)/'result.json'),initial_checkpoint_sha256=sha(meta['config']['checkpoint']),arm=arm,updates=240,learning_rate=.0003,batch='UP_TO_FOUR_DISTINCT_TRAIN_MUSIC_GROUPS_PER_LENGTH',length_cycle=[90,150,240],auxiliary_weights=WEIGHTS,rank_weight=0. if arm=='NoRank' else 1.,rank_temperature=.1,ranking='MASKED_DIAGONAL_STRIDE3_MULTI_POSITIVE_LOGSUMEXP',auxiliary_scope='UNCHANGED_R41_FULL_PARENT_RECONSTRUCTION_AND_REAL_CONTEXT_CLOCK_CROPS',all_arms_forward_identical_candidate_branches=True,selection_uses_val=False,test_tensor_reads=0,can_route=False)
    write(out/'protocol.json',protocol);write(out/'schedule.json',order)
    if replay:
        forbid_optimizers();old=Path(replay);assert order==read(old/'schedule.json');history=read(old/'history.json');checks=[]
        for step in (0,120,239):
            model.load_state_dict(torch.load(old/f'step{step}_pre.pt',map_location='cuda',weights_only=True)['state']);value,parts,_=loss(model,pop,meta,order[step],arm,step);vectors=gradients(parts,model);saved=np.load(old/f'gradient{step}.npz')
            for k,v in vectors.items():np.testing.assert_array_equal(v,saved[k])
            assert float(value.detach())==history[step]['loss']
            assert {k:float(v.detach()) for k,v in parts.items()}==history[step]['parts']
            post=torch.load(old/f'step{step}_post.pt',map_location='cpu',weights_only=True)['state'];assert any(not torch.equal(v.cpu(),post[k]) for k,v in model.state_dict().items());checks.append(step)
        write(out/'result.json',dict(arm=arm,exact_gradient_and_loss_steps=checks,post_update_weights_changed=True,optimizer_steps=0,source_result_sha256=sha(old/'result.json'),can_route=False));return
    save(model,out/'initial.pt',0);opt=torch.optim.AdamW(model.parameters(),lr=.0003);history=[]
    for step,task_ids in enumerate(order):
        opt.zero_grad(set_to_none=True);value,parts,detail=loss(model,pop,meta,task_ids,arm,step)
        if step in (0,120,239):save(model,out/f'step{step}_pre.pt',step);np.savez_compressed(out/f'gradient{step}.npz',**gradients(parts,model))
        value.backward();norm=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
        if not np.isfinite(norm) or norm<=0:raise ValueError('Finite nonzero actual update required')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();history.append(dict(step=step,tasks=task_ids,loss=float(value.detach()),parts={k:float(v.detach()) for k,v in parts.items()},gradient=norm,**detail))
        if step in (0,120,239):save(model,out/f'step{step}_post.pt',step+1)
        if (step+1)%40==0:print('E2_P2_TRAIN',arm,step+1,history[-1]['loss'],flush=True)
    save(model,out/'model.pt',240);write(out/'history.json',history);write(out/'result.json',dict(arm=arm,optimizer_steps=240,initial_digest=initial,final_digest=digest(model),model_parameters=sum(p.numel() for p in model.parameters()),initial_checkpoint_sha256=protocol['initial_checkpoint_sha256'],model_sha256=sha(out/'model.pt'),benchmark_sha256=protocol['benchmark_sha256'],schedule_sha256=sha(out/'schedule.json'),can_route=False,operational_qualified=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.benchmark,a.evaluation,a.arm,a.out,a.replay_from)
