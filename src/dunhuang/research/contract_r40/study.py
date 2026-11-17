"""R39 executor-matched training. Frozen baselines and postprocessing stay untouched."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.bridge_r38.study import runtime,setup_pair
from dunhuang.research.bridge_r38.sampler import sample_bridge,condition,project
from dunhuang.research.bridge_r38 import sampler as parent_sampler
from dunhuang.research.stabilize_r39.projection import continuous_root,Projector
from dunhuang.research.stabilize_r39.study import prepare,sampled
from dunhuang.research.stabilize_r39.smoothing import residuals
from dunhuang.research.coverage_r34.common import population_data,sample,digest,forbid_optimizers,same_arrays
from dunhuang.research.coverage_r34.context_train import checkpoint,restore_context
from dunhuang.research.bridge_r38.source import packet
from dunhuang.research.coverage_r34.context_contract import context
from dunhuang.research.transition_r35.train import targets
from dunhuang.research.transition_r37.evaluate import decode,world_fk
from dunhuang.research.transition_r37.geometry import c2_envelope
from dunhuang.research.control_r33.objective import body_dynamics,error_scales
from dunhuang.research.reference_r26.geometry import fk22


def projection(rt,proposal,anchor,budget,root_budget):
    out=project(rt,proposal,anchor,budget)
    w=torch.as_tensor(c2_envelope(256,[104,152]),device=out.device,dtype=out.dtype)
    out[:,:,4:7]=continuous_root(proposal[:,:,4:7],anchor[:,:,4:7],w,rt.n.legacy.scaler.scale_[4:7].to(out),root_budget)
    return out


def physical_loss(rt,pred,y,a,gap,scales,root_scales,jerk_scale):
    lo,hi=gap;raw=rt.n.unnormalize(pred.clone());truth=rt.n.unnormalize(y.clone())
    p=fk22(raw,rt.s)[:,lo-3:hi+3];q=fk22(truth,rt.s)[:,lo-3:hi+3]
    pose=(pred[:,lo:hi,7:]-y[:,lo:hi,7:]).square().mean()
    geometry=(p-q).square().mean();dyn,_,_=body_dynamics(p,q,torch.ones_like(p[...,0],dtype=torch.bool),scales)
    e=(p-p[:,:,:1])-(q-q[:,:,:1]);jerk=torch.nn.functional.smooth_l1_loss(torch.diff(e,n=3,dim=1)*27000/jerk_scale,torch.zeros_like(e[:,3:]))
    root_error=raw[:,lo-3:hi+3,4:7]-truth[:,lo-3:hi+3,4:7]
    roots=[torch.nn.functional.smooth_l1_loss(torch.diff(root_error,n=k,dim=1)*30**k/root_scales[k],torch.zeros_like(root_error[:,k:])) for k in range(3)]
    value=pose+.2*geometry+.1*dyn+.05*jerk+.1*sum(roots)
    return value,dict(pose=pose,geometry=geometry,dynamic=dyn,jerk=jerk,root_D0=roots[0],root_D1=roots[1],root_D2=roots[2])


def train(cfg,out,arm,replay=None):
    parent=read('src/dunhuang/research/bridge_r38/config.json');rt=runtime(parent)
    restore_context(rt,Path(cfg['r38'])/'training/context.pt');initial=digest(rt.c.context_branch)
    _,_,data,_=population_data();cases=read(Path(cfg['r38'])/'training/training.json')['paired_train_cases']
    if set(cases)&set(cfg['held_cases']):raise ValueError('Held sources entered adaptation')
    cal=read(Path(cfg['r39'])/'calibration/result.json')['parameters'];examples={};cache={};ledger=[]
    # Both arms have the same 16 cases, initialization, batch order, refresh counts and updates.
    for i in range(0,len(cases),4):
        ids=cases[i:i+4];_,_,m=targets(rt,data,ids);base,_,_=sample(rt,m,seeds=[cfg['seed']+j for j in ids]);y,m,a,G,ref,cp=setup_pair(rt,data,ids,parent,base)
        for j,c in enumerate(ids):examples[c]=(y[j:j+1],m[j:j+1],a[j:j+1])
    yall=torch.cat([examples[c][0] for c in cases]);aall=torch.cat([examples[c][2] for c in cases]);lo,hi=parent['gap']
    with torch.no_grad():
        q=fk22(rt.n.unnormalize(yall.clone()),rt.s)[:,lo-3:hi+3];p=fk22(rt.n.unnormalize(aall.clone()),rt.s)[:,lo-3:hi+3]
        scales=error_scales(p,q,torch.ones_like(p[...,0],dtype=torch.bool)).clamp_min(.01)
        e=(p-p[:,:,:1])-(q-q[:,:,:1]);jerk_scale=torch.diff(e,n=3,dim=1).mul(27000).square().mean().sqrt().clamp_min(1.)
        er=rt.n.unnormalize(aall.clone())[:,lo-3:hi+3,4:7]-rt.n.unnormalize(yall.clone())[:,lo-3:hi+3,4:7]
        root_scales=torch.stack([torch.diff(er,n=k,dim=1).mul(30**k).square().mean().sqrt().clamp_min(.01) for k in range(3)])
    write(out/'calibration.json',dict(cases=cases,body_D1_D2=scales.tolist(),body_D3=float(jerk_scale),root_D0_D1_D2=root_scales.tolist(),targets='TRUE_CONTINUOUS_MOTION_NOT_RTS'))
    def assemble(ids):
        y=torch.cat([examples[c][0] for c in ids]);m=torch.cat([examples[c][1] for c in ids]);a=torch.cat([examples[c][2] for c in ids]);raw=rt.n.unnormalize(y.clone()).cpu().numpy();R=np.stack([decode(x)[0] for x in raw]);R[:,lo:hi]=np.nan
        return y,m,a,packet(R,parent['gap'],['paired_'+str(c) for c in ids]),context(y,parent['gap'])
    def refresh(step):
        values={};meta=[]
        for i in range(0,len(cases),4):
            ids=cases[i:i+4];y,m,a,ref,cp=assemble(ids);orig=parent_sampler.project
            if arm=='Matched':parent_sampler.project=Projector(orig,'RootContinuous',cal['root_budget_m'])
            try:pred,nr,states=sample_bridge(rt,m,ref,cp,a,parent,[cfg['seed']+j for j in ids],True)
            finally:parent_sampler.project=orig
            times=[t for t,x in states]
            for j,c in enumerate(ids):values['case'+str(c)]=np.stack([x[j] for t,x in states]);cache[c]=values['case'+str(c)]
            meta.append(dict(cases=ids,noise=nr))
        np.savez_compressed(out/f'cache_{step}.npz',times=times,cases=cases,**values);ledger.append(dict(step=step,weights=digest(rt.c.context_branch),samples=meta));return times
    order=np.random.default_rng(cfg['seed']);schedule=[list(map(int,order.choice(cases,4,replace=False))) for _ in range(cfg['steps'])];write(out/'schedule.json',schedule)
    budget=torch.tensor(parent['budget_rad']*c2_envelope(256,parent['gap']),device='cuda',dtype=torch.float32);weights=np.clip(np.linspace(0,rt.d.guidance_weight*2,50),None,rt.d.guidance_weight)
    history=[];observations=[];refresh_id=0
    if replay:
        forbid_optimizers();expected=read(Path(replay)/'training.json');checkpoint_source=Path(replay);opt=None
    else:opt=torch.optim.AdamW(rt.c.context_branch.parameters(),lr=cfg['learning_rate']);checkpoint(rt,out/'initial.pt',cfg,0)
    def loss(step):
        ids=schedule[step];ti=step%5;t=times[ti];torch.manual_seed(cfg['seed']+step);torch.cuda.manual_seed_all(cfg['seed']+step)
        y,m,a,ref,cp=assemble(ids);xt=torch.tensor(np.stack([cache[c][ti] for c in ids]),device='cuda');ts=torch.full((4,),t,device='cuda',dtype=torch.long)
        if step%3==2:xt=rt.d.q_sample(a,ts,torch.randn_like(a))
        old=condition(rt,ref,cp,parent['gap'])
        try:
            if arm=='Matched':
                _,proposal=rt.d.model_predictions(xt,m,None,ts,weight=float(weights[cfg['cache_indices'][ti]]),clip_x_start=rt.d.clip_denoised)
                pred=projection(rt,proposal,a,budget,cal['root_budget_m']);value,parts=physical_loss(rt,pred,y,a,parent['gap'],scales,root_scales,jerk_scale)
            else:
                proposal=rt.d.model(xt,m,None,ts,cond_drop_prob=0);pred=project(rt,proposal,a,budget)
                pose=(proposal[:,lo:hi,7:]-y[:,lo:hi,7:]).square().mean();root=(pred[:,lo:hi,4:7]-y[:,lo:hi,4:7]).square().mean()
                p=fk22(rt.n.unnormalize(pred.clone()),rt.s)[:,lo-3:hi+3];q=fk22(rt.n.unnormalize(y.clone()),rt.s)[:,lo-3:hi+3];geometry=(p-q).square().mean();old_scales=torch.tensor(read(Path(cfg['r38'])/'training/calibration.json')['scales'],device='cuda');dyn,_,_=body_dynamics(p,q,torch.ones_like(p[...,0],dtype=torch.bool),old_scales)
                value=pose+.2*geometry+.1*dyn+.1*root;parts=dict(pose=pose,geometry=geometry,dynamic=dyn,root=root)
            return value,{k:float(v.detach()) for k,v in parts.items()},t
        finally:rt.c.change.inject=old;rt.c.clear_all()
    for step in range(cfg['steps']):
        if step in cfg['refresh_steps']:
            if replay:restore_context(rt,Path(replay)/f'refresh_{step}.pt')
            else:checkpoint(rt,out/f'refresh_{step}.pt',cfg,step)
            times=refresh(step)
            if replay:same_arrays(out/f'cache_{step}.npz',Path(replay)/f'cache_{step}.npz')
        watch=step in (0,119,239)
        if replay and not watch:continue
        if replay:restore_context(rt,Path(replay)/f'step{step}_pre.pt')
        elif watch:checkpoint(rt,out/f'step{step}_pre.pt',cfg,step)
        rt.c.context_branch.zero_grad(set_to_none=True);value,parts,t=loss(step);torch.use_deterministic_algorithms(True)
        try:value.backward()
        finally:torch.use_deterministic_algorithms(False)
        grad=float(sum(p.grad.square().sum() for p in rt.c.context_branch.parameters() if p.grad is not None).sqrt())
        if not np.isfinite(grad) or grad<=0 or not torch.isfinite(value):raise ValueError('Nonfinite or zero training gradient')
        torch.nn.utils.clip_grad_norm_(rt.c.context_branch.parameters(),1.,error_if_nonfinite=True)
        row=dict(step=step,cases=schedule[step],t=t,loss=float(value.detach()),parts=parts,gradient=grad)
        if replay:
            assert row==next(r for r in expected['observations'] if r['step']==step),row
        else:
            opt.step();history.append(row)
            if watch:checkpoint(rt,out/f'step{step}_post.pt',cfg,step+1)
        if watch:observations.append(row)
        if (step+1)%40==0:print('R40_TRAIN',arm,row,flush=True)
    if replay:final=expected['final_digest']
    else:checkpoint(rt,out/'context.pt',cfg,cfg['steps']);final=digest(rt.c.context_branch)
    rt.c.context_branch.zero_grad(set_to_none=True);rt.check();rt.close()
    write(out/'training.json',dict(arm=arm,optimizer_steps=0 if replay else cfg['steps'],initial_digest=initial,final_digest=final,history=history,observations=observations,cache=ledger,cases=cases,held=cfg['held_cases'],paired_truth_only=True,hidden_gap_truth_in_conditions=False,frozen_source=True,semantic_phase='UNKNOWN',operational_qualified=False,scope='COMBINED_EXECUTION_CONTRACT_NOT_SINGLE_COMPONENT_ATTRIBUTION'))


def generate(cfg,out,training,replay=None):
    parent=read('src/dunhuang/research/bridge_r38/config.json');rt=runtime(parent);restore_context(rt,Path(training)/'context.pt');rt.c.context_branch.requires_grad_(False);forbid_optimizers();_,_,data,_=population_data()
    old=np.load(Path(cfg['r38'])/'evaluation/arrays.npz');r37=np.load(Path(parent['r37'])/'generation/arrays.npz');baseline=np.load(Path(cfg['r39'])/'generation/arrays.npz');meta=read(Path(cfg['r38'])/'evaluation/evaluation.json');cal=read(Path(cfg['r39'])/'calibration/result.json')['parameters'];stats=np.load(Path(cfg['r39'])/'calibration/arrays.npz');arrays=dict(skeleton_J22=old['skeleton_J22'],parents=old['parents']);records=[]
    for row in meta['records']:
        key=row['key'];m,a,ref,cp,target=prepare(rt,parent,data,row,old,r37);anchor=rt.n.unnormalize(a.clone()).cpu().numpy();raw,nr,trace=sampled(rt,m,ref,cp,a,parent,parent['seeds'],'RootContinuous',cal['root_budget_m']);smooth=[];sm=[]
        for p,b in zip(raw,anchor):v,s=residuals(p,b,parent['gap'],cal,stats['root_normalizer_scale'],stats['root_normalizer_offset'],parent['budget_rad']);smooth.append(v);sm.append(s)
        variants=dict(Geometry=anchor,Frozen=baseline[key+'_RootContinuous_139'],Matched=raw,ResidualRTS=np.stack(smooth));arrays[key+'_reference_R']=old[key+'_reference_R'];arrays[key+'_reference22']=old[key+'_reference22'];arrays[key+'_root_trace']=trace
        if target is not None:arrays[key+'_target_139']=target
        for name,v in variants.items():arrays[key+'_'+name+'_139']=v;arrays[key+'_'+name+'_22']=np.stack([world_fk(*decode(p),arrays['skeleton_J22'],arrays['parents']) for p in v])
        records.append(dict(row,smoothing=sm,noise=nr));print('R40_SAMPLE',key,flush=True);np.savez_compressed(out/'arrays.npz',**arrays)
    rt.check();rt.close();result=dict(records=records,optimizer_steps=0,training_sha256=sha(Path(training)/'context.pt'),whole_song_generations=0,native24='NOT_EVALUABLE_22_OF24',operational_qualified=False);write(out/'result.json',result)
    if replay:same_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');assert result==read(Path(replay)/'result.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','generate']);p.add_argument('--arm',choices=['Matched','Extra'],default='Matched');p.add_argument('--out',required=True);p.add_argument('--training');p.add_argument('--replay-from');a=p.parse_args();out=Path(a.out)
    if out.exists():raise ValueError('Fresh output required')
    out.mkdir(parents=True);cfg=read('src/dunhuang/research/contract_r40/config.json');write(out/'protocol.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,mode=a.mode,arm=a.arm));torch.set_num_threads(1)
    if a.mode=='train':train(cfg,out,a.arm,a.replay_from)
    else:generate(cfg,out,a.training,a.replay_from)
if __name__=='__main__':main()
