"""Fresh full sampling; no teacher or hidden motion target is consumed as a condition."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.bridge_r38.study import runtime
from dunhuang.research.coverage_r34.context_train import restore_context
from dunhuang.research.coverage_r34.common import forbid_optimizers,same_arrays
from dunhuang.research.stabilize_r39.smoothing import residuals
from dunhuang.research.transition_r37.evaluate import decode,world_fk
from dunhuang.research.transition_r37.geometry import c2_envelope
from .train import sampled
from .diagnose import measure


def run(data,training,out,label,replay=None,frozen_from=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh generation')
    out.mkdir(parents=True);cfg=read(Path(__file__).with_name('config.json'));parent=read('src/dunhuang/research/bridge_r38/config.json');data=Path(data);z=np.load(data/'arrays.npz');meta=read(data/'result.json');rows=meta['records'];rt=runtime(parent);rt.c.context_branch.requires_grad_(False);forbid_optimizers();cal=read(Path(cfg['r39'])/'calibration/result.json')['parameters'];stats=np.load(Path(cfg['r39'])/'calibration/arrays.npz');J=z['skeleton_J22'];parents=z['parents'];arrays=dict(skeleton_J22=J,parents=parents);records=[];selected=[]
    # All12 ChangE held windows, one metadata-first Train window per source, plus four paired held.
    selected=[dict(r,label=r['key']+' · '+r['role'],evaluation_key=r['key'],target_music=None) for r in rows if r['role']=='held']
    for uid in cfg['change_train_recordings']:
        r=min((r for r in rows if r['recording_uid']==uid),key=lambda r:r.get('start',0));selected.append(dict(r,label=r['key']+' · Train学习对照',evaluation_key=r['key'],target_music=None))
    old=np.load('outputs/transition_r37_02/generation/arrays.npz')
    for uid in cfg['change_train_recordings'][:2]:
        r=min((r for r in rows if r['recording_uid']==uid),key=lambda r:r.get('start',0))
        for name in ('old_failed','new_plan'):selected.append(dict(r,key=r['key']+'__'+name,evaluation_key=r['key'],target_music=name,label=uid+' · 同来源同噪声换音乐 '+name))
    seeds=parent['seeds'];previous=np.load(Path(frozen_from)/'arrays.npz') if frozen_from else None
    if frozen_from:assert read(Path(frozen_from)/'result.json')['data_sha256']==sha(data/'arrays.npz'),'Frozen comparison population changed'
    for row in selected:
        key=row['key'];ek=row['evaluation_key'];rr=dict(row,key=ek);working={name:z[name] for name in (ek+'_anchor',ek+'_music',ek+'_truth')}
        if row['target_music']:working[ek+'_music']=old[row['target_music']+'_music35']
        batch=[rr,rr];anchor=np.repeat(z[ek+'_anchor_raw'][None],2,0);R=decode(z[ek+'_truth'])[0];variants={'Geometry':anchor};noises={}
        for arm,path in [('Frozen',Path(cfg['r38'])/'training/context.pt'),(label,Path(training)/'context.pt')]:
            if arm=='Frozen' and previous is not None:variants[arm]=previous[key+'_Frozen_139'];continue
            restore_context(rt,path);pred,nr,_=sampled(rt,working,batch,parent,seeds,cal);variants[arm]=rt.n.unnormalize(pred.clone()).cpu().numpy();noises[arm]=nr
        variants[label+'RTS']=np.stack([residuals(x,a,[104,152],cal,stats['root_normalizer_scale'],stats['root_normalizer_offset'],.6)[0] for x,a in zip(variants[label],anchor)])
        metrics={};curves={};arrays[key+'_reference_R']=R
        for name,x in variants.items():
            arrays[key+'_'+name+'_139']=x;arrays[key+'_'+name+'_22']=np.stack([world_fk(*decode(v),J,parents) for v in x]);metrics[name],curves[name]=measure(x,R,J,parents,anchor)
        audio=None
        if row['target_music']:
            audio=key+'.wav';shutil.copyfile(Path('outputs/transition_r37_02/generation')/(row['target_music']+'.wav'),out/audio)
        records.append(dict(key=key,label=row['label'],methods=list(variants),seeds=seeds,audio=audio,scope='同录制连续缺口；参考显示真实隐藏目标，仅用于评价，未输入模型。ChangE音乐不配对；身体及相对Root观测，接触UNKNOWN。',role=row['role'],domain=row['domain'],recording_uid=row['recording_uid'],source_interval=row['source_interval'],root_position_observed=row['root_position_observed'],target_music=row['target_music'],metrics=dict(values=metrics,curves=curves),noise=noises))
        np.savez_compressed(out/'arrays.npz',**arrays);print('R42_GENERATE',label,key,flush=True)
    rt.check();rt.close();result=dict(records=records,label=label,training_sha256=sha(Path(training)/'context.pt'),data_sha256=sha(data/'arrays.npz'),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),teacher_reads=0,inference_motion_optimizer_steps=0,optimizer_steps=0,hidden_gap_truth_in_conditions=False,whole_song_generations=0,native24='NOT_EVALUABLE_22_OF24',operational_qualified=False)
    if frozen_from:result['frozen_comparison_arrays_sha256']=sha(Path(frozen_from)/'arrays.npz')
    write(out/'result.json',result)
    if replay:
        same_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');old=read(Path(replay)/'result.json');new=read(out/'result.json');old.pop('code_commit');new.pop('code_commit');assert old==new
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--training',required=True);p.add_argument('--out',required=True);p.add_argument('--label',default='Feasible');p.add_argument('--replay-from');p.add_argument('--frozen-from');a=p.parse_args();run(a.data,a.training,a.out,a.label,a.replay_from,a.frozen_from)
