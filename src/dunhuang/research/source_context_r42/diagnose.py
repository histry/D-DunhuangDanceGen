"""Frozen-weight source/context interventions; hidden and projected effects are saved separately."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.bridge_r38.study import runtime
from dunhuang.research.bridge_r38 import sampler
from dunhuang.research.bridge_r38.source import packet
from dunhuang.research.coverage_r34.common import population_data,forbid_optimizers,same_arrays,digest
from dunhuang.research.coverage_r34.context_train import restore_context
from dunhuang.research.stabilize_r39.study import prepare
from dunhuang.research.stabilize_r39.projection import Projector
from dunhuang.research.stabilize_r39.smoothing import residuals
from dunhuang.research.transition_r37.evaluate import decode,world_fk
from dunhuang.research.transition_r37.geometry import c2_envelope


def measure(raw,reference,J,parents,anchor):
    rows=[];curves=[]
    for x,a in zip(raw,anchor):
        R,root=decode(x);P=relative_fk(R,J,parents);Q=relative_fk(reference,J,parents);e=P-Q;rv=Rotation.from_matrix((reference.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().reshape(256,22);A,_=decode(a);protected=c2_envelope(256,[104,152])==0
        rows.append(dict(gap_body_angle_rad=float(rv[104:152].mean()),gap_shoulder_wrist_rmse_m=float(np.sqrt(np.mean(e[104:152,[13,14,16,17,18,19,20,21]]**2))),gap_body_D2=float(np.sqrt(np.mean((np.diff(e[101:155],n=2,axis=0)*900)**2))),gap_body_D3=float(np.sqrt(np.mean((np.diff(e[101:155],n=3,axis=0)*27000)**2))),root_D2=float(np.sqrt(np.mean((np.diff(root[101:155],n=2,axis=0)*900)**2))),activity=float(np.sqrt(np.mean((np.diff(P[104:152],axis=0)*30)**2))),upper_range_m=float(np.mean(np.ptp(P[104:152,12:],axis=0))),protected_exact=bool(np.array_equal(x[protected],a[protected]))))
        c=np.zeros(256);c[3:]=np.sqrt(np.mean((np.diff(e,n=3,axis=0)*27000)**2,axis=(1,2)));curves.append(c)
    return rows,np.mean(curves,axis=0).tolist()


def intervene(rt,m,ref,cp,a,parent,seeds,mode,budget,cache=False):
    traces=[];state={};oldcond=sampler.condition;oldproject=sampler.project;oldinject=rt.c.inject;oldpred=rt.d.model_predictions
    def condition(rt,source,ctx,gap):
        rt.c.clear_all();rt.c.set_source(source);rt.c.set_context(ctx);old=rt.c.change.inject
        g=torch.tensor(np.ones(256) if mode=='SourceCenter' else 1-c2_envelope(256,gap),device='cuda',dtype=a.dtype)[None,:,None]
        rt.c.change.inject=lambda i,h:h+(old(i,h)-h)*g
        return old
    def predict(*args,**kw):state['t']=int(args[3][0]);return oldpred(*args,**kw)
    def inject(layer,h):
        with_source=rt.c.change.inject(layer,h);ds=with_source-h;dc=rt.c.context_branch.delta(layer,h)
        if mode=='SourceOnly':dc=torch.zeros_like(dc)
        if mode=='ContextOnly':ds=torch.zeros_like(ds)
        row=dict(call=len(traces),t=state.get('t'),layer=layer)
        for name,sl in [('all',slice(None)),('gap',slice(104,152)),('center',slice(116,140))]:
            hv=h[:,sl];sv=ds[:,sl];cv=dc[:,sl];den=hv.norm().clamp_min(1e-9);row[name]=dict(source_ratio=float(sv.norm()/den),context_ratio=float(cv.norm()/den),cosine=float(torch.nn.functional.cosine_similarity(sv.flatten()[None],cv.flatten()[None],eps=1e-9)))
        traces.append(row);return h+dc if mode=='ContextOnly' else with_source+dc
    sampler.condition=condition;sampler.project=Projector(oldproject,'RootContinuous',budget);rt.c.inject=inject;rt.d.model_predictions=predict
    try:result,noise,states=sampler.sample_bridge(rt,m,ref,cp,a,parent,seeds,cache)
    finally:sampler.condition=oldcond;sampler.project=oldproject;rt.c.inject=oldinject;rt.d.model_predictions=oldpred
    return rt.n.unnormalize(result.clone()).cpu().numpy(),noise,traces,states


def run(out,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh P0 diagnostic')
    out.mkdir(parents=True);cfg=read(Path(__file__).with_name('config.json'));parent=read('src/dunhuang/research/bridge_r38/config.json');rt=runtime(parent);restore_context(rt,Path(cfg['r38'])/'training/context.pt');rt.c.context_branch.requires_grad_(False);forbid_optimizers();_,_,data,_=population_data();before=digest(rt.c.context_branch)
    old=np.load(Path(cfg['r38'])/'evaluation/arrays.npz');r37=np.load(Path(parent['r37'])/'generation/arrays.npz');baseline=np.load(Path(cfg['r39'])/'generation/arrays.npz');meta=read(Path(cfg['r38'])/'evaluation/evaluation.json');specs=[r for r in meta['records'] if not r['reference_is_target']];cal=read(Path(cfg['r39'])/'calibration/result.json')['parameters'];stats=np.load(Path(cfg['r39'])/'calibration/arrays.npz');J=old['skeleton_J22'];parents=old['parents'];arrays=dict(skeleton_J22=J,parents=parents);records=[]
    for k,row in enumerate(specs):
        key=row['key'];m,a,ref,cp,target=prepare(rt,parent,data,row,old,r37);anchor=rt.n.unnormalize(a.clone()).cpu().numpy();variants=dict(Geometry=anchor);trace={};noises={};R=old[key+'_reference_R'];curves={};metrics={}
        donor=specs[(k+1)%len(specs)];alternate=old[donor['key']+'_reference_R'].copy();alternate[104:152]=np.nan;donor_ref=packet(np.repeat(alternate[None],len(a),0),parent['gap'],[donor['key']]*len(a))
        other_music=torch.tensor(np.repeat(r37[donor['key']+'_music35'][None],len(a),0),device='cuda')
        for arm in cfg['p0_arms']:
            raw,nr,tt,_=intervene(rt,other_music if arm=='WrongMusic' else m,donor_ref if arm=='WrongSource' else ref,cp,a,parent,parent['seeds'],arm,cal['root_budget_m']);variants[arm]=raw;trace[arm]=tt;noises[arm]=nr
            if arm=='Joint':assert np.array_equal(raw,baseline[key+'_RootContinuous_139']),'Frozen R39 exact replay failed'
            print('R42_P0',key,arm,flush=True)
        variants['JointRTS']=np.stack([residuals(v,b,parent['gap'],cal,stats['root_normalizer_scale'],stats['root_normalizer_offset'],parent['budget_rad'])[0] for v,b in zip(variants['Joint'],anchor)])
        effects={name:dict(max_abs_raw=float(abs(x-variants['Joint']).max()),gap_mean_body_angle_to_joint_rad=float(Rotation.from_matrix((np.stack([decode(v)[0] for v in variants['Joint']]).swapaxes(-1,-2)@np.stack([decode(v)[0] for v in x])).reshape(-1,3,3)).magnitude().reshape(len(x),256,22)[:,104:152].mean())) for name,x in variants.items()}
        arrays[key+'_reference_R']=R
        for name,x in variants.items():
            arrays[key+'_'+name+'_139']=x;arrays[key+'_'+name+'_22']=np.stack([world_fk(*decode(v),J,parents) for v in x]);metrics[name],curves[name]=measure(x,R,J,parents,anchor)
        if row.get('audio'):shutil.copyfile(Path(cfg['r39'])/'generation'/row['audio'],out/row['audio'])
        records.append(dict(key=key,label='R42 冻结条件干预 · '+key,methods=list(variants),seeds=parent['seeds'],audio=row.get('audio'),scope='异源参考距离是保持代理，不是连续真值或文化风格真值；每步有界SO(3)/Root投影',metrics=dict(values=metrics,curves=curves),projected_effects=effects,layer_trace=trace,noise=noises,wrong_source=donor['key'],wrong_music=donor['key'],source_and_context_input='SAME_ORIGINAL_HIDDEN_PARALLEL_RESIDUALS',phase_truth='UNKNOWN'))
        np.savez_compressed(out/'arrays.npz',**arrays)
    assert digest(rt.c.context_branch)==before;rt.check();rt.close();result=dict(records=records,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),optimizer_steps=0,weights_frozen=True,source_gate_original='1-C2_ENVELOPE',source_center_policy='FIXED_RESEARCH_INTERVENTION_ONLY',native24='NOT_EVALUABLE_22_OF24',whole_song_generations=0,operational_qualified=False);write(out/'result.json',result)
    if replay:
        same_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');old=read(Path(replay)/'result.json');new=read(out/'result.json');old.pop('code_commit');new.pop('code_commit');assert old==new
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.out,a.replay_from)
