"""One registered bridge-task candidate. Watchable evaluation precedes SDMR diagnostics."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,subprocess
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from dunhuang.research.coverage_r34.common import Runtime,population_data,sample,digest,forbid_optimizers,same_arrays
from dunhuang.research.coverage_r34.context_train import checkpoint,restore_context
from dunhuang.research.coverage_r34.context_contract import context
from dunhuang.research.transition_r35.train import targets
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.transition_r37.evaluate import decode,metrics,world_fk,curves
from dunhuang.research.transition_r37.geometry import c2_envelope
from dunhuang.research.control_r33.objective import body_dynamics,error_scales
from dunhuang.research.reference_r26.geometry import fk22
from .geometry import observed_bridge,polynomial
from .source import install,packet
from .sampler import sample_bridge,condition,project

def runtime(cfg):
    rt=Runtime(read('src/dunhuang/research/transition_r35/config.json'),context_training=True)
    restore_context(rt,Path(cfg['r36'])/'context/context.pt');install(rt);return rt

def anchor_for(rt,y,base,gap):
    raw=rt.n.unnormalize(y.clone()).cpu().numpy();anchors=[];Rs=[]
    for i,row in enumerate(raw):
        R,_=decode(row);R[gap[0]:gap[1]]=np.nan;G=observed_bridge(R,gap);Rs.append(G)
        x=row.copy();x[:,7:]=G[...,:2,:].reshape(256,132)
        hidden=x.copy();hidden[gap[0]:gap[1],4:7]=np.nan;x[:,4:7]=polynomial(hidden[:,4:7],gap)
        # Unknown contact channels in the gap use the frozen model state, never y.
        x[gap[0]:gap[1],:4]=rt.n.unnormalize(base[i:i+1].clone())[0,gap[0]:gap[1],:4].cpu().numpy()
        anchors.append(x)
    return rt.n.normalize(torch.tensor(np.stack(anchors),device='cuda',dtype=torch.float32)),np.stack(Rs)

def setup_pair(rt,data,ids,cfg,base):
    selected,y,m=targets(rt,data,ids);raw=rt.n.unnormalize(y.clone()).cpu().numpy();rot=np.stack([decode(r)[0] for r in raw]);g=cfg['gap']
    visible=rot.copy();visible[:,g[0]:g[1]]=np.nan;ref=packet(visible,g,['paired_'+str(i) for i in ids]);cp=context(y,g)
    a,G=anchor_for(rt,y,base,g);return y,m,a,G,ref,cp

def train(cfg,out):
    rt=runtime(cfg);_,_,data,z=population_data();initial=digest(rt.c.context_branch)
    cases=np.load('outputs/transition_r35_01/context/initial_states.npz')['cases'].tolist();held=cfg['paired_evaluation_cases']
    if set(cases)&set(held):raise ValueError('Evaluation cases entered bridge training')
    cache={};examples={};cachemeta=[]
    for i in range(0,len(cases),4):
        ids=cases[i:i+4];_,_,music=targets(rt,data,ids);base,nr,_=sample(rt,music,seeds=[cfg['seed']+j for j in ids]);setup=setup_pair(rt,data,ids,cfg,base)
        y,m,anchor,G,ref,cp=setup;pred,nr,states=sample_bridge(rt,m,ref,cp,anchor,cfg,[cfg['seed']+j for j in ids],True)
        times=[t for t,x in states]
        for j,c in enumerate(ids):
            examples[c]=(y[j:j+1],m[j:j+1],anchor[j:j+1],G[j:j+1]);cache[c]=np.stack([x[j] for t,x in states])
        cachemeta.append(dict(cases=ids,noise=nr))
    np.savez_compressed(out/'initial_states.npz',times=times,cases=cases,**{'case'+str(k):v for k,v in cache.items()})
    # Geometry-vs-real dynamics calibrate using this Train population only.
    all_y=torch.cat([examples[c][0] for c in cases]);all_a=torch.cat([examples[c][2] for c in cases])
    with torch.no_grad():
        truth=fk22(rt.n.unnormalize(all_y.clone()),rt.s);geom=fk22(rt.n.unnormalize(all_a.clone()),rt.s)
        scales=error_scales(geom[:,100:156],truth[:,100:156],torch.ones(len(cases),56,22,device='cuda',dtype=torch.bool)).clamp_min(.01)
    write(out/'calibration.json',dict(scales=scales.cpu().tolist(),cases=cases,scope='TRAIN_GEOMETRY_VS_TRUE_D1_D2_ONLY'))
    opt=torch.optim.AdamW(rt.c.context_branch.parameters(),lr=cfg['learning_rate']);history=[];observations=[];rng=np.random.default_rng(cfg['seed'])
    checkpoint(rt,out/'initial.pt',cfg,0);budget=torch.tensor(cfg['budget_rad']*c2_envelope(256,cfg['gap']),device='cuda',dtype=torch.float32)
    for step in range(cfg['steps']):
        ids=[int(x) for x in rng.choice(cases,4,replace=False)];ti=step%len(times);t=times[ti];seed=cfg['seed']+step;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        y=torch.cat([examples[c][0] for c in ids]);m=torch.cat([examples[c][1] for c in ids]);a=torch.cat([examples[c][2] for c in ids]);raw=rt.n.unnormalize(y.clone()).detach().cpu().numpy()
        R=np.stack([decode(r)[0] for r in raw]);R[:,cfg['gap'][0]:cfg['gap'][1]]=np.nan;ref=packet(R,cfg['gap'],['paired_'+str(c) for c in ids]);cp=context(y,cfg['gap'])
        ts=torch.full((4,),t,device='cuda',dtype=torch.long);xt=torch.tensor(np.stack([cache[c][ti] for c in ids]),device='cuda')
        if step%3==2:xt=rt.d.q_sample(a,ts,torch.randn_like(a))
        old=condition(rt,ref,cp,cfg['gap']);opt.zero_grad(set_to_none=True)
        try:
            proposal=rt.d.model(xt,m,None,ts,cond_drop_prob=0);pred=project(rt,proposal,a,budget)
            lo,hi=cfg['gap'];res=proposal[:,lo:hi,7:]-y[:,lo:hi,7:];pose=res.square().mean()
            root=(pred[:,lo:hi,4:7]-y[:,lo:hi,4:7]).square().mean()
            pp=fk22(rt.n.unnormalize(pred.clone()),rt.s)[:,lo-3:hi+3];target=fk22(rt.n.unnormalize(y.clone()),rt.s)[:,lo-3:hi+3]
            geometry=(pp-target).square().mean();dyn,parts,_=body_dynamics(pp,target,torch.ones_like(pp[...,0],dtype=torch.bool),scales)
            value=pose+.2*geometry+cfg['dynamic_weight']*dyn+.1*root
            if step in (0,119,239):checkpoint(rt,out/f'step{step}_pre.pt',cfg,step)
            torch.use_deterministic_algorithms(True)
            try:value.backward()
            finally:torch.use_deterministic_algorithms(False)
            grad=float(torch.sqrt(sum(p.grad.square().sum() for p in rt.c.context_branch.parameters() if p.grad is not None)))
            if not np.isfinite(grad) or grad<=0 or not torch.isfinite(value):raise ValueError('Invalid bridge gradient')
            torch.nn.utils.clip_grad_norm_(rt.c.context_branch.parameters(),1.,error_if_nonfinite=True);opt.step()
            row=dict(step=step,cases=ids,t=t,state='ACTUAL_FROZEN_INITIAL_BRIDGE' if step%3!=2 else 'Q_GEOMETRY_NOT_Q_TRUTH',loss=float(value.detach()),pose=float(pose.detach()),geometry=float(geometry.detach()),dynamic=float(dyn.detach()),root=float(root.detach()))
            history.append(row)
            if step in (0,119,239):observations.append(dict(row,gradient=grad));checkpoint(rt,out/f'step{step}_post.pt',cfg,step+1)
        finally:rt.c.change.inject=old;rt.c.clear_all()
        if (step+1)%40==0:print('R38_BRIDGE_TRAIN',step+1,row,flush=True)
    checkpoint(rt,out/'context.pt',cfg,cfg['steps']);final=digest(rt.c.context_branch);rt.c.context_branch.zero_grad(set_to_none=True);rt.check();rt.close()
    write(out/'training.json',dict(initial_digest=initial,final_digest=final,optimizer_steps=cfg['steps'],history=history,observations=observations,cache=cachemeta,paired_train_cases=cases,evaluation_cases=held,cache_policy='INITIAL_MATCHED_BRIDGE_STATES_NOT_FULLY_ON_POLICY',trainable='EXISTING_CONTEXT_ONLY',ChangE_training=False,FineDance_training=False,hidden_gap_truth_in_conditions=False,operational_qualified=False))

def evaluate(cfg,out,training,replay=None):
    rt=runtime(cfg);forbid_optimizers();_,_,data,z=population_data();arrays=dict(skeleton_J22=rt.s.J[0,:22].cpu().numpy(),parents=np.array(rt.s.parents[:22]));records=[];g=cfg['gap'];seeds=cfg['seeds']
    parent=np.load(Path(cfg['r37'])/'generation/arrays.npz');pmeta=read(Path(cfg['r37'])/'generation/result.json');initial=Path(training)/'initial.pt';trained=Path(training)/'context.pt'
    specifications=[dict(key='paired_'+str(c),case=c,kind='PAIRED_CONTINUOUS_HELD_FROM_THIS_ADAPTATION') for c in cfg['paired_evaluation_cases']]+[dict(key=r['key'],parent=r,kind='CROSS_SOURCE_STRESS_NO_CONTINUOUS_TRUTH') for r in pmeta['records']]
    for spec in specifications:
        key=spec['key'];restore_context(rt,initial)
        if 'case' in spec:
            ids=[spec['case']]*2;_,_,m=targets(rt,data,ids);base,nr,_=sample(rt,m,seeds=seeds);y,m,a,G,ref,cp=setup_pair(rt,data,ids,cfg,base);raw=rt.n.unnormalize(y.clone()).cpu().numpy();R=decode(raw[0])[0];cuts=[g[0],g[1]];audio=None
        else:
            row=spec['parent'];R=parent[key+'_reference_R'];m=torch.tensor(np.repeat(parent[key+'_music35'][None],2,0),device='cuda');base=torch.tensor(parent[key+'_Frozen_normalized'],device='cuda');hidden=R.copy();hidden[g[0]:g[1]]=np.nan;G=np.repeat(observed_bridge(hidden,g)[None],2,0)
            a=base.clone();a[:,:,7:]=torch.tensor(G[...,:2,:].reshape(2,256,132),device='cuda',dtype=torch.float32);ids=np.zeros(256,np.int64)
            for i,e in enumerate(row['binding']['events']):ids[slice(*e['local_interval'])]=i
            ref=packet(np.repeat(hidden[None],2,0),g,[key]*2,np.repeat(ids[None],2,0));cp=context(a,g,partial_body=True);cp.provenance.fill_(2)
            cuts=[e['local_interval'][0] for e in row['binding']['events'][1:]];audio=key+'.wav';__import__('shutil').copyfile(Path(cfg['r37'])/'generation'/audio,out/audio)
        a_raw=rt.n.unnormalize(a.clone()).cpu().numpy();variants={'Geometry':a_raw};noises={}
        for name,path in [('FrozenMatched',initial),('Trained',trained)]:
            restore_context(rt,path);pred,nr,_=sample_bridge(rt,m,ref,cp,a,cfg,seeds);variants[name]=rt.n.unnormalize(pred.clone()).cpu().numpy();noises[name]=nr
            print('R38_EVALUATE',key,name,flush=True)
        arrays[key+'_reference_R']=R;arrays[key+'_reference22']=relative_fk(R,arrays['skeleton_J22'],arrays['parents']);summary={}
        for name,raw in variants.items():
            arrays[key+'_'+name+'_139']=raw;Rs=[];points=[];scores=[]
            for v in raw:
                rr,root=decode(v);Rs.append(rr);points.append(world_fk(rr,root,arrays['skeleton_J22'],arrays['parents']));sc=metrics(rr,R,root,arrays['skeleton_J22'],arrays['parents'],cuts,g,128)
                pp=relative_fk(rr,arrays['skeleton_J22'],arrays['parents']);target=relative_fk(R,arrays['skeleton_J22'],arrays['parents']);lo,hi=g
                sc.update(gap_fk_rmse_m=float(np.sqrt(np.mean((pp[lo:hi]-target[lo:hi])**2))),gap_D1_error=float(np.sqrt(np.mean((np.diff(pp[lo-2:hi+2]-target[lo-2:hi+2],axis=0)*30)**2))),gap_D2_error=float(np.sqrt(np.mean((np.diff(pp[lo-2:hi+2]-target[lo-2:hi+2],n=2,axis=0)*900)**2))))
                scores.append(sc)
            arrays[key+'_'+name+'_R']=np.stack(Rs);arrays[key+'_'+name+'_22']=np.stack(points);summary[name]=scores
        if 'parent' in spec:
            arrays[key+'_R37Joint_22']=parent[key+'_Joint_22'];arrays[key+'_R37Joint_139']=parent[key+'_Joint_139']
        records.append(dict(key=key,kind=spec['kind'],gap=g,cuts=cuts,seeds=seeds,audio=audio,methods=summary,noise=noises,reference_is_target='case' in spec,Root_truth='OBSERVED_PAIRED' if 'case' in spec else 'UNKNOWN',contact_truth='UNKNOWN',binding=spec.get('parent',{}).get('binding'),semantic_phase='UNKNOWN_NOT_TRAINED'))
        np.savez_compressed(out/'arrays.npz',**arrays)
    rt.c.context_branch.zero_grad(set_to_none=True);rt.check();rt.close();result=dict(records=records,optimizer_steps=0,training_sha256=sha(Path(training)/'context.pt'),whole_song_generations=0,operational_qualified=False,native24='NOT_EVALUABLE_22_OF24')
    write(out/'evaluation.json',result)
    if replay:same_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');assert result==read(Path(replay)/'evaluation.json')

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','evaluate']);p.add_argument('--out',required=True);p.add_argument('--training');p.add_argument('--replay-from');a=p.parse_args();out=Path(a.out)
    if out.exists():raise ValueError('Fresh output required')
    out.mkdir(parents=True);cfg=read('src/dunhuang/research/bridge_r38/config.json');write(out/'protocol.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,mode=a.mode))
    torch.set_num_threads(1)
    if a.mode=='train':train(cfg,out)
    else:evaluate(cfg,out,a.training,a.replay_from)
if __name__=='__main__':main()
