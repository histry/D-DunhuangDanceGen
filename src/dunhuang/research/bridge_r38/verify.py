"""Fresh-process training snapshot readouts, zero optimizer calls; output replay is separate."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import numpy as np
import torch
from .study import runtime,setup_pair
from .source import packet
from .sampler import condition,project
from dunhuang.research.coverage_r34.common import population_data,sample,forbid_optimizers
from dunhuang.research.coverage_r34.context_contract import context
from dunhuang.research.coverage_r34.context_train import restore_context
from dunhuang.research.transition_r35.train import targets
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.transition_r37.evaluate import decode
from dunhuang.research.transition_r37.geometry import c2_envelope
from dunhuang.research.reference_r26.geometry import fk22
from dunhuang.research.control_r33.objective import body_dynamics

def run(training,out):
    training=Path(training);cfg=read('src/dunhuang/research/bridge_r38/config.json');rt=runtime(cfg);forbid_optimizers();torch.set_num_threads(1)
    old=read(training/'training.json');cache=np.load(training/'initial_states.npz');times=cache['times'].tolist();scales=torch.tensor(read(training/'calibration.json')['scales'],device='cuda');_,_,data,_=population_data();examples={}
    needed={c for item in old['observations'] for c in item['cases']};cases=cache['cases'].tolist()
    for i in range(0,len(cases),4):
        ids=cases[i:i+4]
        if not needed.intersection(ids):continue
        _,_,music=targets(rt,data,ids);restore_context(rt,training/'initial.pt');base,_,_=sample(rt,music,seeds=[cfg['seed']+j for j in ids]);y,m,a,G,ref,cp=setup_pair(rt,data,ids,cfg,base)
        for j,c in enumerate(ids):examples[c]=(y[j:j+1],m[j:j+1],a[j:j+1])
    results=[];budget=torch.tensor(cfg['budget_rad']*c2_envelope(256,cfg['gap']),device='cuda',dtype=torch.float32)
    for item in old['observations']:
        step=item['step'];ids=item['cases'];t=item['t'];torch.manual_seed(cfg['seed']+step);torch.cuda.manual_seed_all(cfg['seed']+step)
        y=torch.cat([examples[c][0] for c in ids]);m=torch.cat([examples[c][1] for c in ids]);a=torch.cat([examples[c][2] for c in ids]);raw=rt.n.unnormalize(y.clone()).cpu().numpy();R=np.stack([decode(r)[0] for r in raw]);lo,hi=cfg['gap'];R[:,lo:hi]=np.nan
        ref=packet(R,cfg['gap'],['paired_'+str(c) for c in ids]);cp=context(y,cfg['gap']);ts=torch.full((4,),t,device='cuda',dtype=torch.long)
        xt=torch.tensor(np.stack([cache['case'+str(c)][times.index(t)] for c in ids]),device='cuda')
        if step%3==2:xt=rt.d.q_sample(a,ts,torch.randn_like(a))
        restore_context(rt,training/f'step{step}_pre.pt');rt.c.context_branch.zero_grad(set_to_none=True);branch=condition(rt,ref,cp,cfg['gap'])
        try:
            proposal=rt.d.model(xt,m,None,ts,cond_drop_prob=0);pred=project(rt,proposal,a,budget)
            pose=(proposal[:,lo:hi,7:]-y[:,lo:hi,7:]).square().mean();root=(pred[:,lo:hi,4:7]-y[:,lo:hi,4:7]).square().mean()
            pp=fk22(rt.n.unnormalize(pred.clone()),rt.s)[:,lo-3:hi+3];target=fk22(rt.n.unnormalize(y.clone()),rt.s)[:,lo-3:hi+3];geometry=(pp-target).square().mean()
            dyn,_,_=body_dynamics(pp,target,torch.ones_like(pp[...,0],dtype=torch.bool),scales);value=pose+.2*geometry+cfg['dynamic_weight']*dyn+.1*root
            torch.use_deterministic_algorithms(True)
            try:value.backward()
            finally:torch.use_deterministic_algorithms(False)
            gradient=float(torch.sqrt(sum(p.grad.square().sum() for p in rt.c.context_branch.parameters() if p.grad is not None)))
            assert float(value.detach())==item['loss'] and gradient==item['gradient'],(step,float(value.detach()),item['loss'],gradient,item['gradient'])
            results.append(dict(step=step,loss=float(value.detach()),gradient=gradient,exact=True))
        finally:rt.c.change.inject=branch;rt.c.clear_all()
    rt.c.context_branch.zero_grad(set_to_none=True);rt.check();rt.close();write(out,dict(checks=results,optimizer_steps=0,full_training_trajectory_replayed=False,scope='THREE_SAVED_PRE_UPDATE_LOSS_AND_GRADIENT_READOUTS',context_sha256=sha(training/'context.pt')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.training,a.out)
