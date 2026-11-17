"""Frozen R38 weights; Train-only scale registration, then six fixed real sampling cases."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.bridge_r38.study import runtime,setup_pair
from dunhuang.research.bridge_r38 import sampler as parent_sampler
from dunhuang.research.bridge_r38.source import packet
from dunhuang.research.bridge_r38.geometry import observed_bridge
from dunhuang.research.coverage_r34.common import population_data,sample,forbid_optimizers,same_arrays,digest
from dunhuang.research.coverage_r34.context_train import restore_context
from dunhuang.research.coverage_r34.context_contract import context
from dunhuang.research.transition_r35.train import targets
from dunhuang.research.transition_r37.evaluate import decode,world_fk
from dunhuang.research.transition_r37.geometry import c2_envelope
from .projection import Projector
from .smoothing import residuals

def start(cfg):
    parent=read('src/dunhuang/research/bridge_r38/config.json');rt=runtime(parent);restore_context(rt,Path(cfg['r38'])/'training/context.pt');rt.c.context_branch.requires_grad_(False);forbid_optimizers();return parent,rt

def sampled(rt,m,ref,cp,a,parent,seeds,mode,budget_m):
    original=parent_sampler.project;projection=Projector(original,mode,budget_m);parent_sampler.project=projection
    try:pred,noise,_=parent_sampler.sample_bridge(rt,m,ref,cp,a,parent,seeds)
    finally:parent_sampler.project=original
    return rt.n.unnormalize(pred.clone()).cpu().numpy(),noise,np.stack(projection.trace)

def calibrate(cfg,out,replay=None):
    parent,rt=start(cfg);_,_,data,_=population_data();before=digest(rt.c.context_branch);train=read(Path(cfg['r38'])/'training/training.json')['paired_train_cases']
    if set(train)&set(parent['paired_evaluation_cases']):raise ValueError('Held cases entered calibration')
    arrays=dict(skeleton_J22=rt.s.J[0,:22].cpu().numpy(),parents=np.array(rt.s.parents[:22]));truth=[];geometry=[];predicted=[];ledger=[]
    for i in range(0,len(train),4):
        ids=train[i:i+4];_,_,m=targets(rt,data,ids);seeds=[cfg['calibration_seed']+c for c in ids];base,_,_=sample(rt,m,seeds=seeds)
        y,m,a,G,ref,cp=setup_pair(rt,data,ids,parent,base);raw,noise,trace=sampled(rt,m,ref,cp,a,parent,seeds,'Raw',1.)
        truth.extend(rt.n.unnormalize(y.clone()).cpu().numpy());geometry.extend(rt.n.unnormalize(a.clone()).cpu().numpy());predicted.extend(raw);ledger.append(dict(cases=ids,seeds=seeds,noise=noise));print('R39_CALIBRATION',i+len(ids),flush=True)
    T=np.stack(truth);B=np.stack(geometry);P=np.stack(predicted);a,b=parent['gap'];w=c2_envelope(256,parent['gap']);support=w[a:b]>=.25
    roots=np.linalg.norm(T[:,a:b,4:7]-B[:,a:b,4:7],axis=-1)[:,support]/w[a:b][support][None]
    root_sigma=float(np.sqrt(np.mean((P[:,a:b,4:7]-T[:,a:b,4:7])**2)));root_q=float(np.sqrt(np.mean((np.diff(T[:,a:b,4:7]-B[:,a:b,4:7],n=3,axis=1)*27000)**2)))
    angle_errors=[];jerk_errors=[]
    for t,g,p in zip(T,B,P):
        R,_=decode(t);G,_=decode(g);Q,_=decode(p);error=Rotation.from_matrix((R[a:b].swapaxes(-1,-2)@Q[a:b]).reshape(-1,3,3)).as_rotvec();angle_errors.append(error)
        v=[]
        for rot in (R,G):v.append(Rotation.from_matrix((rot[:-1].swapaxes(-1,-2)@rot[1:]).reshape(-1,3,3)).as_rotvec().reshape(255,22,3)*30)
        jerk_errors.append(np.diff(v[0][a:b-1]-v[1][a:b-1],n=2,axis=0)*900)
    body_sigma=float(np.sqrt(np.mean(np.concatenate(angle_errors)**2)));body_q=float(np.sqrt(np.mean(np.stack(jerk_errors)**2)))
    def bounded(v,key):return float(np.clip(v,*cfg[key]))
    cal=dict(root_budget_m=bounded(np.quantile(roots,cfg['root_budget_quantile']),'root_budget_range_m'),root_observation_sigma_m=bounded(root_sigma,'root_observation_sigma_range_m'),body_observation_sigma_rad=bounded(body_sigma,'body_observation_sigma_range_rad'),root_process_jerk_m_s3=bounded(root_q,'root_process_jerk_range_m_s3'),body_process_jerk_rad_s3=bounded(body_q,'body_process_jerk_range_rad_s3'))
    arrays.update(truth139=T,geometry139=B,raw139=P,root_normalizer_scale=rt.n.legacy.scaler.scale_[4:7].cpu().numpy(),root_normalizer_offset=rt.n.legacy.scaler.min_[4:7].cpu().numpy());np.savez_compressed(out/'arrays.npz',**arrays)
    assert digest(rt.c.context_branch)==before;rt.check();rt.close();result=dict(parameters=cal,unclipped_estimates=dict(root_sigma_m=root_sigma,body_sigma_rad=body_sigma,root_process_jerk_m_s3=root_q,body_process_jerk_rad_s3=body_q),cases=train,ledger=ledger,training_checkpoint_sha256=sha(Path(cfg['r38'])/'training/context.pt'),optimizer_steps=0,scope=cfg['calibration_policy'],regularization_not_calibrated_probability=True)
    write(out/'result.json',result)
    if replay:same_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');assert result==read(Path(replay)/'result.json')

def prepare(rt,parent,data,row,old,r37):
    key=row['key'];g=parent['gap'];seeds=parent['seeds']
    if row['reference_is_target']:
        ids=[int(key.split('_')[1])]*2;_,_,m=targets(rt,data,ids);base,_,_=sample(rt,m,seeds=seeds);y,m,a,G,ref,cp=setup_pair(rt,data,ids,parent,base);target=rt.n.unnormalize(y.clone()).cpu().numpy()
    else:
        R=old[key+'_reference_R'];hidden=R.copy();hidden[g[0]:g[1]]=np.nan;G=np.repeat(observed_bridge(hidden,g)[None],2,0);base=torch.tensor(r37[key+'_Frozen_normalized'],device='cuda');a=base.clone();a[:,:,7:]=torch.tensor(G[...,:2,:].reshape(2,256,132),device='cuda',dtype=torch.float32)
        m=torch.tensor(np.repeat(r37[key+'_music35'][None],2,0),device='cuda');ids=np.zeros(256,np.int64)
        for i,e in enumerate(row['binding']['events']):ids[slice(*e['local_interval'])]=i
        ref=packet(np.repeat(hidden[None],2,0),g,[key]*2,np.repeat(ids[None],2,0));cp=context(a,g,partial_body=True);cp.provenance.fill_(2);target=None
    anchor=rt.n.unnormalize(a.clone()).cpu().numpy()
    if not np.array_equal(anchor,old[key+'_Geometry_139']):raise ValueError('Anchor differs from frozen R38')
    return m,a,ref,cp,target

def generate(cfg,root,out,replay=None):
    parent,rt=start(cfg);_,_,data,_=population_data();before=digest(rt.c.context_branch);original=Path(cfg['r38']);old=np.load(original/'evaluation/arrays.npz');r37=np.load(Path(parent['r37'])/'generation/arrays.npz');meta=read(original/'evaluation/evaluation.json');cal=read(Path(root)/'calibration/result.json')['parameters'];stats=np.load(Path(root)/'calibration/arrays.npz');arrays=dict(skeleton_J22=old['skeleton_J22'],parents=old['parents']);records=[]
    for row in meta['records']:
        key=row['key'];m,a,ref,cp,target=prepare(rt,parent,data,row,old,r37);anchor=rt.n.unnormalize(a.clone()).cpu().numpy();variants={'Geometry':anchor};traces={};noises={}
        for mode in ('Raw','RootContinuous'):
            raw,noise,trace=sampled(rt,m,ref,cp,a,parent,parent['seeds'],mode,cal['root_budget_m']);variants[mode]=raw;traces[mode]=trace;noises[mode]=noise
            if mode=='Raw' and not np.array_equal(raw,old[key+'_Trained_139']):raise ValueError('Raw arm does not replay frozen R38')
            print('R39_SAMPLE',key,mode,flush=True)
        values=[];smoothing=[]
        for raw,base in zip(variants['RootContinuous'],anchor):
            v,s=residuals(raw,base,parent['gap'],cal,stats['root_normalizer_scale'],stats['root_normalizer_offset'],parent['budget_rad']);values.append(v);smoothing.append(s)
        variants['ResidualRTS']=np.stack(values);arrays[key+'_reference_R']=old[key+'_reference_R'];arrays[key+'_reference22']=old[key+'_reference22']
        if target is not None:arrays[key+'_target_139']=target
        for mode,raw in variants.items():
            arrays[key+'_'+mode+'_139']=raw;arrays[key+'_'+mode+'_22']=np.stack([world_fk(*decode(v),arrays['skeleton_J22'],arrays['parents']) for v in raw])
        for mode,trace in traces.items():arrays[key+'_'+mode+'_root_projection_trace']=trace
        if row['audio']:shutil.copyfile(original/'evaluation'/row['audio'],out/row['audio'])
        records.append(dict(row,smoothing=smoothing,noise=noises,raw_r38_bitwise_replay=True));np.savez_compressed(out/'arrays.npz',**arrays)
    assert digest(rt.c.context_branch)==before;rt.check();rt.close();result=dict(records=records,optimizer_steps=0,weights_frozen=True,whole_song_generations=0,operational_qualified=False,root_continuous_applied='EACH_X0_BEFORE_DDIM_EPSILON_RECOMPUTATION',calibration_sha256=sha(Path(root)/'calibration/result.json'),native24='NOT_EVALUABLE_22_OF24')
    write(out/'result.json',result)
    if replay:same_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');assert result==read(Path(replay)/'result.json')

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibrate','generate']);p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();cfg=read('src/dunhuang/research/stabilize_r39/config.json');out=Path(a.out)
    if out.exists():raise ValueError('Fresh output required')
    out.mkdir(parents=True);write(out/'protocol.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,mode=a.mode));torch.set_num_threads(1)
    if a.mode=='calibrate':calibrate(cfg,out,a.replay_from)
    else:generate(cfg,a.root,out,a.replay_from)
if __name__=='__main__':main()
