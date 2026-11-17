"""Only Context learns. Same initialization, real sampler caches, fixed budget and Train-only teachers."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.bridge_r38.study import runtime
from dunhuang.research.bridge_r38 import sampler
from dunhuang.research.bridge_r38.source import packet
from dunhuang.research.coverage_r34.context_contract import context
from dunhuang.research.coverage_r34.context_train import checkpoint,restore_context
from dunhuang.research.coverage_r34.common import digest,same_arrays,forbid_optimizers
from dunhuang.research.contract_r40.study import projection
from dunhuang.research.stabilize_r39.projection import Projector
from dunhuang.research.paired_event_r41.train import seeded
from pytorch3d.transforms import rotation_6d_to_matrix
from .teacher import fk,temporal_loss
from dunhuang.research.transition_r37.geometry import c2_envelope
from dunhuang.research.transition_r37.geometry import c2_envelope
from dunhuang.research.transition_r37.geometry import c2_envelope
from dunhuang.research.transition_r37.geometry import c2_envelope
from dunhuang.research.transition_r37.geometry import c2_envelope


def assemble(rt,z,rows):
    keys=[r['key'] for r in rows];a=torch.tensor(np.stack([z[k+'_anchor'] for k in keys]),device='cuda');m=torch.tensor(np.stack([z[k+'_music'] for k in keys]),device='cuda');raw=np.stack([z[k+'_truth'] for k in keys]);hidden=rotation_6d_to_matrix(torch.tensor(raw[:,:,7:].reshape(len(raw),256,22,6),device='cuda')).cpu().numpy();hidden[:,104:152]=np.nan;ref=packet(hidden,[104,152],keys);cp=context(a,[104,152]);cp.valid[:,:,:4]=False;cp.values[:,:,:4]=float('nan')
    for j,r in enumerate(rows):
        if not r['root_position_observed']:cp.valid[j,:,4:7]=False;cp.values[j,:,4:7]=float('nan')
    cp.validate();return a,m,ref,cp


def sampled(rt,z,rows,parent,seeds,cal,cache=False):
    a,m,ref,cp=assemble(rt,z,rows);original=sampler.project;sampler.project=Projector(original,'RootContinuous',cal['root_budget_m'])
    try:x,nr,states=sampler.sample_bridge(rt,m,ref,cp,a,parent,seeds,cache)
    finally:sampler.project=original
    return x,nr,states


def run(data,teacher,out,arm,replay=None):
    out=Path(out)
    approval=Path(data).parent/'receipts'/('p0_browser_acceptance.json' if arm=='FeasibleChangE' else 'main_browser_acceptance.json')
    if not approval.exists() or not read(approval).get('accepted'):raise ValueError('Actual watchable delivery must precede this training phase')
    if out.exists():raise ValueError('Fresh training output')
    out.mkdir(parents=True);cfg=read(Path(__file__).with_name('config.json'));parent=read('src/dunhuang/research/bridge_r38/config.json');data=Path(data);teacher=Path(teacher);meta=read(data/'result.json');z=np.load(data/'arrays.npz');teach=np.load(teacher/'arrays.npz');rows={r['key']:r for r in meta['records'] if r['role']=='train'};teacher_keys=teach['keys'].tolist();assert set(rows)==set(teacher_keys);targets={k:teach['teacher139'][j] for j,k in enumerate(teacher_keys)};paired=[k for k,r in rows.items() if r['domain']=='AIST++'];change=[k for k,r in rows.items() if r['domain']=='ChangE'];rng=np.random.default_rng(cfg['seed']);schedule=[]
    for step in range(cfg['steps']):schedule.append(list(map(str,rng.choice(paired,4,replace=False))) if arm=='RealAIST' else list(map(str,rng.choice(change,2,replace=False)))+list(map(str,rng.choice(paired,2,replace=False))))
    write(out/'schedule.json',schedule);write(out/'protocol.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,arm=arm,data_sha256=sha(data/'arrays.npz'),teacher_sha256=sha(teacher/'arrays.npz'),trainable='CONTEXT_ONLY_SAME_268160_PARAMETERS',change_music='ZERO_NO_PAIRING_CLAIM',hidden_gap_truth_in_conditions=False,operational_qualified=False));rt=runtime(parent);restore_context(rt,Path(cfg['r38'])/'training/context.pt');initial=digest(rt.c.context_branch);cal=read(Path(cfg['r39'])/'calibration/result.json')['parameters'];bs=torch.tensor(teach['body_scales'],device='cuda');rs=torch.tensor(teach['root_scales'],device='cuda');J=torch.tensor(z['skeleton_J22'],device='cuda');parents=z['parents'];w=torch.tensor(.6*c2_envelope(256,[104,152]),device='cuda',dtype=torch.float32);history=[];watch=[];cache={};ledger=[];times=None;steps=[0,cfg['steps']//2,cfg['steps']-1];pool=paired if arm=='RealAIST' else sorted(rows)
    if replay:forbid_optimizers();expected=read(Path(replay)/'training.json');opt=None
    else:opt=torch.optim.AdamW(rt.c.context_branch.parameters(),lr=cfg['learning_rate']);checkpoint(rt,out/'initial.pt',cfg,0)
    weights=np.clip(np.linspace(0,rt.d.guidance_weight*2,50),None,rt.d.guidance_weight)
    for step in range(cfg['steps']):
        if step in cfg['refresh_steps']:
            if replay:restore_context(rt,Path(replay)/f'refresh_{step}.pt')
            else:checkpoint(rt,out/f'refresh_{step}.pt',cfg,step)
            states_out={};noises=[]
            for j in range(0,len(pool),4):
                keys=pool[j:j+4];rr=[rows[k] for k in keys];x,nr,states=sampled(rt,z,rr,parent,[cfg['seed']+sorted(rows).index(k) for k in keys],cal,True);times=[t for t,v in states]
                for b,k in enumerate(keys):cache[k]=np.stack([v[b] for t,v in states]);states_out[k]=cache[k]
                noises.append(dict(keys=keys,noise=nr))
            np.savez_compressed(out/f'cache_{step}.npz',times=times,**states_out);ledger.append(dict(step=step,weights=digest(rt.c.context_branch),noise=noises))
            if replay:same_arrays(out/f'cache_{step}.npz',Path(replay)/f'cache_{step}.npz')
            print('R42_CACHE',arm,step,len(pool),flush=True)
        if replay and step not in steps:continue
        if replay:restore_context(rt,Path(replay)/f'step{step}_pre.pt')
        elif step in steps:checkpoint(rt,out/f'step{step}_pre.pt',cfg,step)
        keys=schedule[step];rr=[rows[k] for k in keys];a,m,ref,cp=assemble(rt,z,rr);ti=step%5;t=times[ti];seeded(cfg['seed']+step);xt=torch.tensor(np.stack([cache[k][ti] for k in keys]),device='cuda');ts=torch.full((len(keys),),t,device='cuda',dtype=torch.long)
        if step%3==2:xt=rt.d.q_sample(a,ts,torch.randn_like(a))
        target_raw=torch.tensor(np.stack([targets[k] if arm=='FeasibleChangE' else z[k+'_truth'] for k in keys]),device='cuda');root_valid=torch.tensor([rows[k]['root_position_observed'] for k in keys],device='cuda');target_norm=rt.n.normalize(target_raw.clone());old=sampler.condition(rt,ref,cp,[104,152]);rt.c.context_branch.zero_grad(set_to_none=True)
        try:
            _,proposal=rt.d.model_predictions(xt,m,None,ts,weight=float(weights[[0,10,25,40,49][ti]]),clip_x_start=rt.d.clip_denoised);pred=projection(rt,proposal,a,w,cal['root_budget_m']);raw=rt.n.unnormalize(pred.clone());R=rotation_6d_to_matrix(raw[:,:,7:].reshape(len(keys),256,22,6));T=rotation_6d_to_matrix(target_raw[:,:,7:].reshape(len(keys),256,22,6));dynamic,parts=temporal_loss(fk(R,J,parents)-fk(T,J,parents),raw[:,:,4:7]-target_raw[:,:,4:7],root_valid,bs,rs);pose=(proposal[:,104:152,7:]-target_norm[:,104:152,7:]).square().mean();value=pose+.1*dynamic;parts=dict(parts,proposal_body_pose=pose);torch.use_deterministic_algorithms(True)
            try:value.backward()
            finally:torch.use_deterministic_algorithms(False)
            grad=float(sum(p.grad.square().sum() for p in rt.c.context_branch.parameters() if p.grad is not None).sqrt());assert np.isfinite(grad) and grad>0 and torch.isfinite(value);torch.nn.utils.clip_grad_norm_(rt.c.context_branch.parameters(),1.,error_if_nonfinite=True);record=dict(step=step,keys=keys,t=t,loss=float(value.detach()),gradient=grad,parts={k:float(v.detach()) for k,v in parts.items()},root_position_observed=root_valid.cpu().tolist(),hidden_condition_tokens=0)
            if replay:assert record==next(r for r in expected['observations'] if r['step']==step)
            else:
                opt.step();history.append(record)
                if step in steps:checkpoint(rt,out/f'step{step}_post.pt',cfg,step+1)
            if step in steps:watch.append(record)
        finally:rt.c.change.inject=old;rt.c.clear_all()
        if (step+1)%40==0:print('R42_TRAIN',arm,step+1,float(value.detach()),flush=True)
    if replay:final=expected['final_digest']
    else:checkpoint(rt,out/'context.pt',cfg,cfg['steps']);final=digest(rt.c.context_branch)
    rt.c.context_branch.zero_grad(set_to_none=True);rt.check();rt.close();write(out/'training.json',dict(arm=arm,initial_digest=initial,final_digest=final,optimizer_steps=0 if replay else cfg['steps'],history=history,observations=watch,cache=ledger,train_keys=pool,unique_train_keys=len(set(k for r in schedule for k in r)),train_recordings=sorted({rows[k]['recording_uid'] for r in schedule for k in r}),supervision='TRAIN_FEASIBLE_TEACHER' if arm=='FeasibleChangE' else 'TRUE_CONTINUOUS_MOTION',change_music_paired=False,contact_loss=False,whole_song_generations=0,operational_qualified=False));print('R42_TRAIN_DONE',arm,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--teacher',required=True);p.add_argument('--out',required=True);p.add_argument('--arm',choices=['FeasibleChangE','RealChangE','RealAIST'],required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.data,a.teacher,a.out,a.arm,a.replay_from)
