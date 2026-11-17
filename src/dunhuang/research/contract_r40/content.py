"""Shadow content training with a clock that does not imply equal local indices."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.bridge_r38.content import OrderedRetrieval,schedule,relations,multi_positive_loss
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.coverage_r34.common import forbid_optimizers,same_arrays
from .crops import crop_origins,clock_labels,offset_scores,tie_accuracy


def run(data,out,replay=None):
    data=Path(data);out=Path(out)
    if out.exists():raise ValueError('Fresh content directory')
    out.mkdir(parents=True);cfg=read('src/dunhuang/research/contract_r40/config.json');manifest=read(data/'manifest.json');rows=manifest['rows'];normal=read(data/'normalization.json')['statistics'];seed=38041;N=cfg['content_crop_frames'];offsets=cfg['content_offsets'];cache={}
    if sha(data/'normalization.json')!=manifest['normalizer_sha256']:raise ValueError('Train normalizer changed')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);model=OrderedRetrieval().cuda();order=schedule(rows,seed,cfg['content_steps']);write(out/'schedule.json',order)
    write(out/'protocol.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,population_sha256=sha(data/'manifest.json'),normalizer_sha256=sha(data/'normalization.json'),absolute_origins_input=False,labels_only='a+i=b+j',local_support=[30,60],new_blind_test=False,semantic_truth='UNKNOWN'))
    def unit(i):
        if i not in cache:
            p=Path(rows[i]['path'])
            if sha(p)!=rows[i]['sha256']:raise ValueError('Population unit changed')
            z=np.load(p);result={}
            for name in ('music','motion'):
                mask=z[name+'_mask'];v=(z[name]-np.asarray(normal[name]['mean']))/np.asarray(normal[name]['std']);result[name]=np.where(mask,v,0).astype(np.float32);result[name+'_mask']=mask.copy()
            cache[i]=result
        return cache[i]
    def batch(ids,origins=None):
        length=max(rows[i]['frames'] for i in ids) if origins is None else N;m=np.zeros((len(ids),length,50),np.float32);d=np.zeros((len(ids),length,274),np.float32);mask=np.zeros_like(d,bool);valid=np.zeros((len(ids),length),bool)
        for j,i in enumerate(ids):
            z=unit(i);a,b=(0,0) if origins is None else origins[j];n=rows[i]['frames'] if origins is None else N
            if max(a,b)+n>rows[i]['frames']:raise ValueError('Crop outside real unit')
            m[j,:n]=z['music'][a:a+n];d[j,:n]=z['motion'][b:b+n];mask[j,:n]=z['motion_mask'][b:b+n];valid[j,:n]=z['music_mask'][a:a+n].any(-1)
        return [torch.tensor(x,device='cuda') for x in (m,d,mask,valid)]
    if replay:forbid_optimizers();model.load_state_dict(torch.load(Path(replay)/'model.pt',map_location='cuda',weights_only=True)['state']);opt=None
    else:opt=torch.optim.AdamW(model.parameters(),lr=.0002)
    history=[];origin_ledger=[]
    if opt:
        for step,ids in enumerate(order):
            origins=crop_origins(seed,step,len(ids));M,D,mt,dt,rec=model.encode(*batch(ids));p,n=relations([rows[i] for i in ids]);glob=multi_positive_loss(M@D.T,p,n)
            mb,xb,mask,valid=batch(ids,origins);_,_,mt,dt,rec=model.encode(mb,xb,mask,valid);losses=[]
            for j,(a,b) in enumerate(origins):
                ii,jj=clock_labels(N,int(a),int(b));good=valid[j,ii]&mask[j,jj].any(-1);ii=torch.tensor(ii,device='cuda')[good];jj=torch.tensor(jj,device='cuda')[good]
                if not len(ii):raise ValueError('No common observed crop interior')
                # All motion keys are observed genuine tokens; no fixed offset marker is supplied.
                logits=mt[j,ii]@dt[j].T/.1;logits=logits.masked_fill(~mask[j].any(-1)[None],-torch.inf);losses.append(torch.nn.functional.cross_entropy(logits,jj))
            clock=torch.stack(losses).mean();reconstruction=(rec-xb[:,:,:198]).square()[mask[:,:,:198]].mean();value=glob+.1*clock+.1*reconstruction
            opt.zero_grad(set_to_none=True);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();history.append(dict(step=step,loss=float(value.detach()),global_retrieval=float(glob.detach()),independent_clock=float(clock.detach()),reconstruction=float(reconstruction.detach())));origin_ledger.append(origins.tolist())
            if (step+1)%20==0:print('R40_CONTENT_TRAIN',history[-1],flush=True)
        torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},population_sha256=sha(data/'manifest.json')),out/'model.pt');write(out/'history.json',history);write(out/'crop_origins_labels_only.json',origin_ledger)
    val=[i for i,r in enumerate(rows) if r['role']=='val'];donors={i:next(j for j in val if rows[j]['audio_sha256']!=rows[i]['audio_sha256']) for i in val};models={}
    models['R40']=model.eval();old=OrderedRetrieval().cuda().eval();old.load_state_dict(torch.load('outputs/bridge_r38_01/content/model.pt',map_location='cuda',weights_only=True)['state']);models['FrozenR38']=old;arrays=dict(val_indices=np.array(val),offsets=np.array(offsets));metrics={}
    with torch.no_grad():
        for name,net in models.items():
            metrics[name]={}
            for control in ('actual','music_zero','both_zero','music_swap'):
                score=np.zeros((len(val),5,5));token=np.zeros((len(val),5));chance=[]
                for s in range(0,len(val),8):
                    ids=val[s:s+8]
                    for k,delta in enumerate(offsets):
                        origins=np.array([[15+delta,15]]*len(ids));m,x,mask,valid=batch(ids,origins)
                        if control=='music_swap':m,_,_,_=batch([donors[i] for i in ids],origins)
                        if control in ('music_zero','both_zero'):m.zero_()
                        if control=='both_zero':x.zero_()
                        # One shared validity mask across all controls and offsets avoids missingness shortcuts.
                        for j,i in enumerate(ids):
                            mm=unit(i)['music_mask'][:120].all(0)&unit(donors[i])['music_mask'][:120].all(0)
                            dm=unit(i)['motion_mask'][:120].all(0)
                            if not mm.any() or not dm[:-2].any():raise ValueError('No common genuine support')
                            m[j,:,~torch.tensor(mm,device='cuda')]=0;x[j,:,~torch.tensor(dm,device='cuda')]=0
                            mask[j]=torch.tensor(dm,device='cuda')[None].expand(N,-1);valid[j]=True
                        _,_,mt,dt,_=net.encode(m,x,mask,valid)
                        for j,i in enumerate(ids):
                            a=mt[j].cpu().numpy();b=dt[j].cpu().numpy();score[s+j,k]=offset_scores(a,b,offsets);ii,jj=clock_labels(N,15+delta,15);v=a[ii]@b.T;token[s+j,k]=tie_accuracy(v,jj).mean()
                truth=np.broadcast_to(np.arange(5),(len(val),5));acc=tie_accuracy(score,truth);arrays[name+'_'+control+'_scores']=score;arrays[name+'_'+control+'_token']=token
                groups={g:float(acc[[rows[i]['source_group']==g for i in val]].mean()) for g in sorted({rows[i]['source_group'] for i in val})}
                metrics[name][control]=dict(queries=len(val)*5,offset_accuracy=float(acc.mean()),token_clock_recall_at1=float(token.mean()),token_uniform_chance=1/N,offset_uniform_chance=.2,groups=groups);print('R40_CONTENT',name,control,metrics[name][control]['offset_accuracy'],flush=True)
    global_metrics={};p,n=relations([rows[i] for i in val])
    with torch.no_grad():
        for name,net in models.items():
            M=[];D=[]
            for k in range(0,len(val),8):
                m,d,*_=net.encode(*batch(val[k:k+8]));M.append(m.cpu().numpy());D.append(d.cpu().numpy())
            scores=np.concatenate(M)@np.concatenate(D).T;arrays[name+'_global_scores']=scores;eligible=p|n;hits=[]
            for j,i in enumerate(val):
                if not p[j].any() or not n[j].any():continue
                pool=np.flatnonzero(eligible[j]);win=pool[np.isclose(scores[j,pool],scores[j,pool].max(),atol=1e-7,rtol=0)];hits.append(float(p[j,win].mean()))
            global_metrics[name]=dict(queries=len(hits),multi_positive_full_unit_recall_at1=float(np.mean(hits)))
    arrays['global_positive']=p;arrays['global_negative']=n
    np.savez_compressed(out/'evaluation.npz',**arrays);result=dict(metrics=metrics,global_metrics=global_metrics,optimizer_steps=0 if replay else cfg['content_steps'],parameters=sum(p.numel() for p in model.parameters()),units=len(val),population_sha256=sha(data/'manifest.json'),can_route=False,operational_qualified=False,independent_crop_origins=True,global_task='UNCHANGED_MULTI_POSITIVE_FULL_UNIT_RETRIEVAL',synthetic_clock_not_semantic_event_truth=True);write(out/'result.json',result)
    if replay:same_arrays(out/'evaluation.npz',Path(replay)/'evaluation.npz');expected=read(Path(replay)/'result.json');expected['optimizer_steps']=0;assert result==expected
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.data,a.out,a.replay_from)
