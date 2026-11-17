"""Frozen R38 content/position diagnostics on equal real support; no extra training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.bridge_r38.content import OrderedRetrieval
from dunhuang.research.coverage_r34.common import forbid_optimizers

OFFSETS=(-15,-8,0,8,15)
CONTROLS=('actual','music_zero','both_zero','music_swap')

def support(length):
    if length<60:raise ValueError('Actual shared non-circular offset support')
    ticks=np.arange(15,length-15)
    return ticks,np.stack([ticks+d for d in OFFSETS])

def run(data,checkpoint,out,replay=None):
    data=Path(data);out=Path(out)
    if out.exists():raise ValueError('Fresh diagnostics directory')
    out.mkdir(parents=True);manifest=read(data/'manifest.json');normal=read(data/'normalization.json')['statistics'];rows=manifest['rows'];val=[i for i,r in enumerate(rows) if r['role']=='val']
    if sha(data/'normalization.json')!=manifest['normalizer_sha256']:raise ValueError('Normalizer changed')
    expected=next(r['sha256'] for r in read('outputs/bridge_r38_01/artifacts_manifest.json')['files'] if r['path']=='content/model.pt')
    if sha(checkpoint)!=expected:raise ValueError('Frozen R38 checkpoint changed')
    torch.manual_seed(39041);torch.cuda.manual_seed_all(39041);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);model=OrderedRetrieval().cuda().eval();saved=torch.load(checkpoint,map_location='cuda',weights_only=True)
    if saved['population_sha256']!=sha(data/'manifest.json'):raise ValueError('Checkpoint/population binding differs')
    model.load_state_dict(saved['state']);forbid_optimizers();cache={};donors={}
    def unit(i):
        if i not in cache:
            p=Path(rows[i]['path'])
            if sha(p)!=rows[i]['sha256']:raise ValueError('Population unit changed')
            z=np.load(p);cache[i]={k:z[k].copy() for k in z.files}
        return cache[i]
    for i in val:
        options=[j for j in val if rows[j]['frames']==rows[i]['frames'] and rows[j]['audio_sha256']!=rows[i]['audio_sha256']]
        if not options:raise ValueError('No equal-length swapped content control')
        donors[i]=options[0]
    def batch(ids,delta,control):
        n=max(rows[i]['frames']-30 for i in ids);M=np.zeros((len(ids),n,50),np.float32);D=np.zeros((len(ids),n,274),np.float32);mask=np.zeros_like(D,bool);valid=np.zeros((len(ids),n),bool)
        for b,i in enumerate(ids):
            original=unit(i);z=unit(donors[i]) if control=='music_swap' else original;ticks,all_indices=support(rows[i]['frames']);mi=all_indices[OFFSETS.index(delta)];N=len(ticks)
            mm=np.all(np.stack([z['music_mask'][v] for v in all_indices]),axis=0)
            music=(z['music'][mi]-np.asarray(normal['music']['mean']))/np.asarray(normal['music']['std']);motion=(original['motion'][ticks]-np.asarray(normal['motion']['mean']))/np.asarray(normal['motion']['std'])
            observed=original['motion_mask'][ticks];M[b,:N]=np.where(mm,music,0);D[b,:N]=np.where(observed,motion,0);mask[b,:N]=observed;valid[b,:N]=mm.any(-1)&observed.any(-1)
            if control in ('music_zero','both_zero'):M[b,:N]=0
            if control=='both_zero':D[b,:N]=0
        return [torch.tensor(v,device='cuda') for v in (M,D,mask,valid)]
    arrays=dict(val_indices=np.array(val),offsets=np.array(OFFSETS));summary={};groups={}
    with torch.no_grad():
        for control in CONTROLS:
            scores=np.zeros((len(val),len(OFFSETS)));local_accuracy=np.zeros(len(val));local_chance=np.zeros(len(val));coverage=np.zeros(len(val),int)
            for start in range(0,len(val),8):
                ids=val[start:start+8]
                for k,delta in enumerate(OFFSETS):
                    M,D,mask,valid=batch(ids,delta,control);_,_,mt,dt,_=model.encode(M,D,mask,valid)
                    for j,i in enumerate(ids):
                        index=torch.where(valid[j]&mask[j].any(-1))[0][::4];values=mt[j,index]@dt[j,index].T;scores[start+j,k]=float(values.diag().mean())
                        if delta==0:
                            local_accuracy[start+j]=float((values.argmax(-1)==torch.arange(len(index),device='cuda')).float().mean());local_chance[start+j]=1/len(index);coverage[start+j]=len(index)
            center=OFFSETS.index(0);maxima=scores.max(1,keepdims=True);ties=np.isclose(scores,maxima,atol=1e-7,rtol=0);top=(ties[:,center]/ties.sum(1)).mean();comparisons=[]
            for k in range(len(OFFSETS)):
                if k==center:continue
                diff=scores[:,center]-scores[:,k];comparisons.append(float(np.mean((diff>1e-7)+.5*(abs(diff)<=1e-7))))
            per={}
            for group in sorted({rows[i]['source_group'] for i in val}):
                ix=np.array([rows[i]['source_group']==group for i in val]);per[group]=dict(queries=int(ix.sum()),offset_top1_tie_uniform=float((ties[ix,center]/ties[ix].sum(1)).mean()),token_index_recall_at1=float(local_accuracy[ix].mean()))
            arrays[control+'_scores']=scores;arrays[control+'_local_accuracy']=local_accuracy;arrays[control+'_sampled_token_counts']=coverage
            summary[control]=dict(queries=len(val),offset_top1_tie_uniform=float(top),correct_vs_offsets_tie_half=float(np.mean(comparisons)),token_index_recall_at1=float(local_accuracy.mean()),token_uniform_chance=float(local_chance.mean()),groups=per)
            print('R39_CONTENT_DIAGNOSTIC',control,summary[control]['offset_top1_tie_uniform'],summary[control]['token_index_recall_at1'],flush=True)
    for control in CONTROLS[1:]:
        if not np.array_equal(arrays[control+'_sampled_token_counts'],arrays['actual_sampled_token_counts']):raise ValueError('Unequal support between content controls')
    np.savez_compressed(out/'arrays.npz',**arrays);result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),checkpoint_sha256=sha(checkpoint),population_sha256=sha(data/'manifest.json'),offsets_frames=list(OFFSETS),metrics=summary,donor_unit_uids={rows[i]['unit_uid']:rows[j]['unit_uid'] for i,j in donors.items()},support_policy='REAL_MUSIC[t+delta]_AND_MOTION[t]; t=15..N-16; EQUAL_MASK_INTERSECTION_ACROSS_ALL_OFFSETS',ground_truth='PHYSICAL_RECORDING_CLOCK_NOT_ARTISTIC_EVENT_CORRESPONDENCE',validation_scope='HISTORICALLY_EXPOSED_VAL',optimizer_steps=0,can_route=False)
    write(out/'result.json',result)
    if replay:
        same=np.load(Path(replay)/'arrays.npz')
        for k in arrays:assert np.array_equal(arrays[k],same[k]),k
        assert result==read(Path(replay)/'result.json')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.data,a.checkpoint,a.out,a.replay_from)
