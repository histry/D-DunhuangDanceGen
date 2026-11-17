"""Shadow-only ordered retrieval pilot; no routing or generation path changes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,collections,subprocess
from pathlib import Path
import numpy as np
import torch
from .content import OrderedRetrieval,schedule,relations,multi_positive_loss
from dunhuang.research.complete_r36.core import read,write,sha

def run(data,out,replay=None):
    data=Path(data);out=Path(out)
    if out.exists():raise ValueError('Fresh retrieval output')
    out.mkdir(parents=True);manifest=read(data/'manifest.json');rows=manifest['rows'];normalization=read(data/'normalization.json')['statistics']
    if sha(data/'normalization.json')!=manifest['normalizer_sha256']:raise ValueError('Normalizer changed')
    seed=38041;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=OrderedRetrieval().cuda();cache={};energy={};order=schedule(rows,seed,80);np.save(out/'schedule.npy',order)
    def unit(i):
        if i not in cache:
            p=Path(rows[i]['path'])
            if sha(p)!=rows[i]['sha256']:raise ValueError('Source unit changed')
            a=np.load(p);cache[i]={k:a[k].copy() for k in a.files}
        return cache[i]
    def batch(ids,mode='correct'):
        n=max(rows[i]['frames'] for i in ids);m=np.zeros((len(ids),n,50),np.float32);x=np.zeros((len(ids),n,274),np.float32);mm=np.zeros_like(m,bool);xm=np.zeros_like(x,bool)
        for j,i in enumerate(ids):
            z=unit(i);N=rows[i]['frames']
            for name,dest,maskdest in [('music',m,mm),('motion',x,xm)]:
                valid=z[name+'_mask'];v=(z[name]-np.asarray(normalization[name]['mean']))/np.asarray(normalization[name]['std']);dest[j,:N]=np.where(valid,v,0);maskdest[j,:N]=valid
            if mode=='shuffle':
                p=np.random.default_rng(seed+i).permutation(N);m[j,:N]=m[j,p];mm[j,:N]=mm[j,p]
            elif mode=='shift':
                # Cropped interior supports true non-circular +15-frame intervention.
                m[j,:N-15]=m[j,15:N];mm[j,:N-15]=mm[j,15:N];m[j,N-15:]=0;mm[j,N-15:]=False
        return [torch.tensor(v,device='cuda') for v in (m,x,xm,mm.any(-1))]
    history=[]
    if replay:
        from dunhuang.research.state_r27.common import forbid_optimizers
        forbid_optimizers();model.load_state_dict(torch.load(Path(replay)/'model.pt',map_location='cuda',weights_only=True)['state']);opt=None
    else:opt=torch.optim.AdamW(model.parameters(),lr=.0002)
    if opt:
        for step,ids in enumerate(order):
            m,x,mask,valid=batch(ids);M,D,mt,dt,rec=model.encode(m,x,mask,valid);p,n=relations([rows[i] for i in ids]);glob=multi_positive_loss(M@D.T,p,n)
            # Physical same-recording clock auxiliary, not semantic event correspondence.
            local=[]
            for j in range(len(ids)):
                index=torch.where(valid[j]&mask[j].any(-1))[0][::4]
                scores=mt[j,index]@dt[j,index].T/.1;local.append(torch.nn.functional.cross_entropy(scores,torch.arange(len(index),device='cuda')))
            clock=torch.stack(local).mean();reconstruction=(rec-x[:,:,:198]).square()[mask[:,:,:198]].mean();loss=glob+.1*clock+.1*reconstruction
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            history.append(dict(step=step,loss=float(loss.detach()),retrieval=float(glob.detach()),physical_clock_auxiliary=float(clock.detach()),reconstruction=float(reconstruction.detach()),min_actual_positives=int(p.sum(1).min())))
            if (step+1)%20==0:print('R38_ORDERED_CONTENT',history[-1],flush=True)
        torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},population_sha256=sha(data/'manifest.json')),out/'model.pt');write(out/'history.json',history)
    # Energy matching parameters are fixed from Train only; Val does not calibrate bins.
    for i in range(len(rows)):
        z=unit(i);energy[i]=(float(np.sqrt(np.mean(z['music'][z['music_mask']]**2))),float(np.sqrt(np.mean(z['motion'][:,198:264]**2))))
    ee=np.array([energy[i] for i,r in enumerate(rows) if r['role']=='train']);mean=ee.mean(0);std=ee.std(0).clip(.0001)
    val=[i for i,r in enumerate(rows) if r['role']=='val'];model.eval();embeddings={}
    with torch.no_grad():
        for mode in ('correct','shuffle','shift'):
            mm=[];dd=[]
            for s in range(0,len(val),8):
                M,D,*_=model.encode(*batch(val[s:s+8],mode));mm.append(M.cpu().numpy());dd.append(D.cpu().numpy())
            embeddings[mode]=(np.concatenate(mm),np.concatenate(dd))
    p,n=relations([rows[i] for i in val]);length=np.array([rows[i]['frames'] for i in val]);energyval=(np.array([energy[i] for i in val])-mean)/std;metrics={};arrays=dict(positive=p,negative=n,val_indices=np.array(val))
    for mode,(M,D) in embeddings.items():
        scores=M@D.T;arrays[mode+'_scores']=scores;arrays[mode+'_music']=M;arrays[mode+'_motion']=D
        totals={};groups=collections.defaultdict(list)
        for scope in ('all','same_music','same_length_near_energy'):
            hits=[];chance=[];pergroup=collections.defaultdict(list)
            for j,i in enumerate(val):
                eligible=p[j]|n[j];same=np.array([rows[k]['audio_sha256']==rows[i]['audio_sha256'] for k in val])
                if scope=='same_music':eligible &= same
                if scope=='same_length_near_energy':eligible &= (length==rows[i]['frames']) & (abs(energyval[j,0]-energyval[:,1])<=1.)
                if not (p[j]&eligible).any() or not (n[j]&eligible).any():continue
                k=np.flatnonzero(eligible)[np.argmax(scores[j,eligible])];hit=bool(p[j,k]);hits.append(hit);chance.append(float((p[j]&eligible).sum()/eligible.sum()));pergroup[rows[i]['source_group']].append(hit)
            totals[scope]=dict(queries=len(hits),coverage=len(hits)/len(val),recall_at_1=float(np.mean(hits)) if hits else None,uniform_candidate_chance=float(np.mean(chance)) if chance else None,groups={g:dict(queries=len(h),recall_at_1=float(np.mean(h))) for g,h in pergroup.items()})
        metrics[mode]=totals
    np.savez_compressed(out/'evaluation.npz',**arrays);result=dict(optimizer_steps=0 if replay else 80,parameters=sum(p.numel() for p in model.parameters()),population_sha256=sha(data/'manifest.json'),normalizer_sha256=sha(data/'normalization.json'),actual_positive_counts_min=2,batch_size=8,paired_audio_groups_per_batch=2,ordered_music=True,clock_duration_input_excluded=True,metrics=metrics,energy_train_mean=mean.tolist(),energy_train_std=std.tolist(),exposed_validation=True,can_route=False,operational_qualified=False,ablation_training_runs=0)
    write(out/'result.json',result)
    if replay:
        a=np.load(out/'evaluation.npz');b=np.load(Path(replay)/'evaluation.npz')
        for k in a.files:assert np.array_equal(a[k],b[k]),k
        old=read(Path(replay)/'result.json');old['optimizer_steps']=0;assert result==old;assert np.array_equal(np.load(out/'schedule.npy'),np.load(Path(replay)/'schedule.npy'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.data,a.out,a.replay_from)
