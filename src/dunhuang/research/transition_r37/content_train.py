"""One fixed main retrieval pilot after first media delivery; no shuffled-training sweep."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,collections,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.complete_r36.core import read,write,sha
from .content import build_model,relations,multi_positive_loss

def run(data,out,replay=None):
    data=Path(data);out=Path(out)
    if out.exists():raise ValueError('Fresh content training directory')
    out.mkdir(parents=True);manifest=read(data/'manifest.json');cfg=manifest['config'];rows=manifest['rows'];normalization=read(data/'normalization.json')['statistics']
    assert sha(data/'normalization.json')==manifest['normalizer_sha256']
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    device='cuda';model=build_model(width=cfg['width']).to(device);rng=np.random.default_rng(cfg['seed']);cache={}
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if r['role']=='train':groups[r['audio_sha256']].append(i)
        elif r['role']!='val':raise ValueError('Test forbidden')
    def disjoint(i,j):
        A=rows[i]['audio_frame_interval'];B=rows[j]['audio_frame_interval']
        return max(A[0],B[0])>=min(A[1],B[1])
    query_options={k:[i for i in v if any(disjoint(i,j) for j in v)] for k,v in groups.items()}
    groups={k:v for k,v in groups.items() if query_options[k]}
    if len(groups)<cfg['batch_music_groups']:raise ValueError('Insufficient trusted multi-time music groups')
    def batch(ids):
        n=max(rows[i]['frames'] for i in ids);m=np.zeros((len(ids),n,50),np.float32);x=np.zeros((len(ids),n,274),np.float32);mm=np.zeros_like(m,bool);xm=np.zeros_like(x,bool)
        for j,i in enumerate(ids):
            if i not in cache:
                p=Path(rows[i]['path'])
                if sha(p)!=rows[i]['sha256']:raise ValueError('Population unit changed')
                z=np.load(p);cache[i]={k:z[k].copy() for k in z.files}
            z=cache[i];N=rows[i]['frames']
            for name,dest,mask_dest in [('music',m,mm),('motion',x,xm)]:
                valid=z[name+'_mask'];v=(z[name]-np.asarray(normalization[name]['mean']))/np.asarray(normalization[name]['std'])
                dest[j,:N]=np.where(valid,v,0);mask_dest[j,:N]=valid
        return [torch.tensor(v,device=device) for v in (m,x,xm,mm.any(-1))]
    schedule=[]
    for step in range(cfg['updates']):
        ids=[]
        for ak in rng.choice(sorted(groups),cfg['batch_music_groups'],replace=False):
            options=groups[ak];i=int(rng.choice(query_options[ak]));A=rows[i]['audio_frame_interval']
            alternatives=[j for j in options if disjoint(i,j)]
            if not alternatives:raise ValueError('No same-music nonoverlapping competition')
            ids.extend([i,int(rng.choice(alternatives))])
        schedule.append(ids)
    np.save(out/'schedule.npy',np.array(schedule));history=[]
    if replay:
        torch.optim.AdamW=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Replay optimizer forbidden'))
        checkpoint=torch.load(Path(replay)/'model.pt',map_location=device,weights_only=True);model.load_state_dict(checkpoint['state']);optimizer=None
    else:optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
    if optimizer:
        for step,ids in enumerate(schedule):
            model.train();music,x,mask,valid=batch(ids);p,n=relations([rows[i] for i in ids]);scores,rec=model(music,x,mask,valid)
            retrieval=multi_positive_loss(scores,p,n);target=x[:,:,:198];observed=mask[:,:,:198]
            reconstruction=(rec-target).square()[observed].mean();loss=retrieval+cfg['reconstruction_weight']*reconstruction
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            history.append(dict(step=step,loss=float(loss),retrieval=float(retrieval),reconstruction=float(reconstruction)))
            if step%20==0:print('R37_CONTENT_TRAIN',history[-1],flush=True)
        torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},manifest_sha256=sha(data/'manifest.json')),out/'model.pt')
        write(out/'history.json',history)
    model.eval();val=[i for i,r in enumerate(rows) if r['role']=='val'];music_embeddings=[];motion_embeddings=[]
    with torch.no_grad():
        for start in range(0,len(val),8):
            ids=val[start:start+8];music,x,mask,valid=batch(ids);d,_=model.motion_tokens(x,mask);m=model.music(music);m=(m*valid[...,None]).sum(1)/valid.sum(1)[:,None];m=torch.nn.functional.normalize(m,dim=-1)
            music_embeddings.append(m.cpu().numpy());motion_embeddings.append(d.cpu().numpy())
    M=np.concatenate(music_embeddings);D=np.concatenate(motion_embeddings);p,n=relations([rows[i] for i in val]);score=M@D.T;recalls=[];time_recalls=[];examples=[]
    for i,idx in enumerate(val):
        valid=p[i]|n[i]
        if not p[i].any() or not n[i].any():continue
        order=np.flatnonzero(valid)[np.argsort(-score[i,valid],kind='stable')];recalls.append(bool(p[i,order[0]]))
        same=np.array([rows[j]['audio_sha256']==rows[idx]['audio_sha256'] for j in val]);eligible=valid&same
        if (n[i]&same).any():
            local=np.flatnonzero(eligible)[np.argsort(-score[i,eligible],kind='stable')];time_recalls.append(bool(p[i,local[0]]))
        if len(examples)<12:examples.append(dict(query=rows[idx]['unit_uid'],ranked=[dict(unit_uid=rows[val[j]]['unit_uid'],score=float(score[i,j]),verified_clock_positive=bool(p[i,j])) for j in order[:5]]))
    np.savez_compressed(out/'evaluation.npz',music=M,motion=D,scores=score,positive=p,negative=n,val_indices=np.array(val))
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),population_sha256=sha(data/'manifest.json'),
                optimizer_steps=0 if replay else cfg['updates'],parameters=sum(v.numel() for v in model.parameters()),val_units=len(val),
                recall_at_1=float(np.mean(recalls)),same_music_time_recall_at_1=float(np.mean(time_recalls)) if time_recalls else None,
                validation_scope=cfg['validation_scope'],candidate_scope='ALL_VERIFIED_VAL_UNITS_EXCLUDING_AMBIGUOUS_OVERLAPS',
                shuffled_training_control_done=False,energy_matched_control_done=False,examples=examples,can_route=False,operational_qualified=False)
    write(out/'result.json',result)
    if replay:
        a=np.load(out/'evaluation.npz');b=np.load(Path(replay)/'evaluation.npz')
        for k in a.files:assert np.array_equal(a[k],b[k]),k
        assert np.array_equal(np.load(out/'schedule.npy'),np.load(Path(replay)/'schedule.npy'))
        old=read(Path(replay)/'result.json');old['optimizer_steps']=0;assert old==result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.data,a.out,a.replay_from)
