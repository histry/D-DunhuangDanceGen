"""Finite learnability checks before population training. No Val model selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter1d
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.transition_r37.content import multi_positive_loss
from dunhuang.research.coverage_r34.common import forbid_optimizers,same_arrays
from dunhuang.research.contract_r40.crops import tie_accuracy
from .model import EventContent,local_objective,scores_for_offsets
from .population import Population


def seeded(seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)


def digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def initial(cfg):
    seeded(cfg['seed']);model=EventContent(cfg['width']).cuda();path=Path('outputs/bridge_r38_01/content/model.pt');expected=next(r['sha256'] for r in read('outputs/bridge_r38_01/artifacts_manifest.json')['files'] if r['path']=='content/model.pt');assert sha(path)==expected;saved=torch.load(path,map_location='cuda',weights_only=True);missing=model.load_state_dict(saved['state'],strict=False)
    if missing.unexpected_keys or any(not k.startswith('local_') for k in missing.missing_keys):raise ValueError('Explicit global-model initialization')
    return model


def save(model,path,step):torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},step=step),path)


def synthetic(seed,batch,origins=None,zero=False):
    # Known shared random temporal signals, two fixed modality transforms, no human motion claims.
    transform=np.random.default_rng(41100);A=transform.normal(size=(8,50))/np.sqrt(8);B=transform.normal(size=(8,274))/np.sqrt(8);rng=np.random.default_rng(seed);latent=gaussian_filter1d(rng.normal(size=(batch,150,8)),.65,axis=1)
    if origins is None:origins=rng.integers(0,16,(batch,2))
    m=[];d=[]
    for i,(a,b) in enumerate(origins):m.append(latent[i,a:a+90]@A);d.append(latent[i,b:b+90]@B)
    m=torch.tensor(np.stack(m),device='cuda',dtype=torch.float32);d=torch.tensor(np.stack(d),device='cuda',dtype=torch.float32)
    if zero:m.zero_();d.zero_()
    return m,d,torch.ones_like(d,dtype=torch.bool),torch.ones(m.shape[:2],device='cuda',dtype=torch.bool),np.asarray(origins)


def synthetic_evaluate(model,cfg,out):
    model.eval();scores=[];zeros=[];offsets=cfg['offset_candidates']
    with torch.no_grad():
        for k in range(32):
            origins=np.array([[15+delta,15] for delta in offsets]);m,x,mask,valid,_=synthetic(cfg['positive_eval_seed']+k,5,origins);mt,dt,*_=model.content(m,x,mask,valid);scores.append(scores_for_offsets(mt,dt,offsets).cpu().numpy())
            m.zero_();x.zero_();mt,dt,*_=model.content(m,x,mask,valid);zeros.append(scores_for_offsets(mt,dt,offsets).cpu().numpy())
    # Each trial has its own latent signal but controlled injected labels, never equal-index labels.
    a=np.stack(scores);z=np.stack(zeros);truth=np.broadcast_to(np.arange(5),(32,5));result=dict(actual_accuracy=float(tie_accuracy(a,truth).mean()),both_zero_accuracy=float(tie_accuracy(z,truth).mean()),trials=160,scope='SYNTHETIC_SHARED_SIGNAL_HELD_RANDOM_REALIZATIONS_NOT_MUSIC_DANCE_SEMANTICS')
    np.savez_compressed(out/'synthetic_eval.npz',scores=a,both_zero=z,truth=truth);return result


def clock_evaluate(model,pop,ids,cfg,out,label):
    model.eval();offsets=cfg['offset_candidates'];values={};metrics={}
    with torch.no_grad():
        for control in ('actual','music_zero','both_zero'):
            scores=np.zeros((len(ids),5,5))
            for start in range(0,len(ids),8):
                part=ids[start:start+8]
                for k,d in enumerate(offsets):
                    inputs=pop.batch(part,np.array([[15+d,15]]*len(part)),control);mt,dt,*_=model.content(*inputs);scores[start:start+len(part),k]=scores_for_offsets(mt,dt,offsets).cpu().numpy()
            truth=np.broadcast_to(np.arange(5),(len(ids),5));metrics[control]=float(tie_accuracy(scores,truth).mean());values[control]=scores
    np.savez_compressed(out/(label+'_eval.npz'),ids=ids,**values);return metrics


def gradient_readout(parts,model):
    params=tuple(model.music.parameters())+tuple(model.motion.parameters());vectors={}
    for key in ('global','clock','reconstruction'):
        g=torch.autograd.grad(parts[key],params,retain_graph=True,allow_unused=True);vectors[key]=torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(params,g)])
    norms={k:float(v.norm()) for k,v in vectors.items()};cos={a+'__'+b:float(F.cosine_similarity(vectors[a][None],vectors[b][None],eps=1e-10)) for a,b in (('global','clock'),('clock','reconstruction'),('global','reconstruction'))};return dict(norms=norms,cosines=cos)


def objective(model,pop,ids,cfg,step):
    m,x,mask,valid=pop.batch(ids);M,D,_,_,full_rec=model.encode(m,x,mask,valid);p,n=pop.relations(ids);glob=multi_positive_loss(M@D.T,p,n)
    reconstruction=(full_rec-x[:,:,:198]).square()[mask[:,:,:198]].mean()
    rng=np.random.default_rng(cfg['seed']+step);origins=rng.integers(0,16,(len(ids),2));other=origins.copy()
    for i,(a,b) in enumerate(origins):
        options=[t for t in range(31) if abs(t-int(b))<=15 and t!=a];other[i,0]=rng.choice(options)
    mt,dt,*_=model.content(*pop.batch(ids,origins));ca,la,ea=local_objective(mt,dt,origins)
    mt,dt,*_=model.content(*pop.batch(ids,other));cb,lb,eb=local_objective(mt,dt,other)
    expected=torch.tensor(other[:,0]-origins[:,0],device=ea.device,dtype=torch.float32);equiv=F.smooth_l1_loss((eb-ea)/15,expected/15)
    parts=dict(global_=glob,clock=(ca+cb)/2,local=(la+lb)/2,reconstruction=reconstruction,equivariance=equiv);parts['global']=parts.pop('global_');value=sum(cfg['loss_weights'][k]*v for k,v in parts.items())
    return value,parts,dict(origins=origins.tolist(),second_origins=other.tolist(),reconstruction_observed_values=int(mask[:,:,:198].sum()),full_reconstruction_frames=int(sum(pop.rows[i]['frames'] for i in ids)))


def fit(model,pop,order,cfg,out,checkpoint_name):
    initial_hash=digest(model);save(model,out/'initial.pt',0);opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']);history=[];snapshots=[];model.train()
    for step,ids in enumerate(order):
        seeded(cfg['seed']+step);opt.zero_grad(set_to_none=True);value,parts,metadata=objective(model,pop,ids,cfg,step);watch=step in (0,len(order)//2,len(order)-1)
        readout=gradient_readout(parts,model) if watch else None
        if watch:save(model,out/f'step{step}_pre.pt',step)
        value.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
        if not np.isfinite(grad) or grad<=0 or not torch.isfinite(value):raise ValueError('Invalid actual gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();row=dict(step=step,ids=ids,loss=float(value.detach()),parts={k:float(v.detach()) for k,v in parts.items()},gradient=grad,**metadata);history.append(row)
        if watch:save(model,out/f'step{step}_post.pt',step+1);snapshots.append(dict(step=step,gradient_tasks=readout))
        if (step+1)%40==0:print('R41_FIT',checkpoint_name,step+1,float(value.detach()),flush=True)
    save(model,out/(checkpoint_name+'.pt'),len(order));write(out/'history.json',history);write(out/'gradient_tasks.json',snapshots);write(out/'schedule.json',order)
    return dict(initial_digest=initial_hash,final_digest=digest(model),optimizer_steps=len(order),unique_train_events=len({i for ids in order for i in ids}),parameters=sum(p.numel() for p in model.parameters()),scope='FULL_CORE_RECONSTRUCTION_AND_GLOBAL_RETRIEVAL_PLUS_POSITION_FREE_LOCAL_TIMING')


def checks(cfg,pop,out,replay=None):
    if replay:
        forbid_optimizers();old=read(Path(replay)/'result.json');model=initial(cfg);model.load_state_dict(torch.load(Path(replay)/'synthetic.pt',map_location='cuda',weights_only=True)['state']);synthetic_result=synthetic_evaluate(model,cfg,out);same_arrays(out/'synthetic_eval.npz',Path(replay)/'synthetic_eval.npz')
        model=initial(cfg);model.load_state_dict(torch.load(Path(replay)/'train_fit/fit.pt',map_location='cuda',weights_only=True)['state']);fit_result=clock_evaluate(model,pop,old['fit_ids'],cfg,out,'fit');same_arrays(out/'fit_eval.npz',Path(replay)/'fit_eval.npz');assert synthetic_result==old['synthetic'] and fit_result==old['train_fit_metrics'];write(out/'result.json',dict(synthetic=synthetic_result,train_fit_metrics=fit_result,optimizer_steps=0,admitted_main=old['admitted_main']));return
    model=initial(cfg);opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']);initial_hash=digest(model);history=[]
    for step in range(cfg['positive_control_steps']):
        seeded(cfg['seed']+step);m,x,mask,valid,origins=synthetic(41000+step,8);mt,dt,*_=model.content(m,x,mask,valid);clock,token,_=local_objective(mt,dt,origins);value=clock+token;opt.zero_grad(set_to_none=True);value.backward();grad=float(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None).sqrt());torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();history.append(dict(step=step,loss=float(value.detach()),gradient=grad))
        if (step+1)%40==0:print('R41_SYNTHETIC',step+1,float(value.detach()),flush=True)
    save(model,out/'synthetic.pt',cfg['positive_control_steps']);write(out/'synthetic_history.json',history);synthetic_result=synthetic_evaluate(model,cfg,out);fit_ids=pop.fit_population();fit_out=out/'train_fit';fit_out.mkdir();model=initial(cfg);fit_training=fit(model,pop,pop.schedule(cfg['real_train_fit_steps'],cfg['seed'],fit_ids),cfg,fit_out,'fit');fit_result=clock_evaluate(model,pop,fit_ids,cfg,out,'fit')
    admitted=synthetic_result['actual_accuracy']>=cfg['positive_gate_accuracy'] and synthetic_result['both_zero_accuracy']<=cfg['zero_control_ceiling'] and fit_result['actual']>=cfg['real_fit_gate_accuracy'] and fit_result['both_zero']<=cfg['zero_control_ceiling']
    result=dict(synthetic=synthetic_result,synthetic_initial_digest=initial_hash,synthetic_optimizer_steps=cfg['positive_control_steps'],train_fit_metrics=fit_result,fit_training=fit_training,fit_ids=fit_ids,fit_population=[{k:pop.rows[i][k] for k in ('event_uid','recording_uid','audio_sha256','role')} for i in fit_ids],admitted_main=bool(admitted),gates=dict(synthetic_accuracy=cfg['positive_gate_accuracy'],fit_accuracy=cfg['real_fit_gate_accuracy'],zero_ceiling=cfg['zero_control_ceiling']),gate_data_role='SYNTHETIC_HELD_REALIZATIONS_AND_REAL_TRAIN_ONLY_NO_VAL',operational_qualified=False);write(out/'result.json',result);print('R41_LEARNABILITY',synthetic_result,fit_result,'MAIN_ADMITTED',admitted,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['checks','main']);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--checks');p.add_argument('--replay-from');a=p.parse_args();out=Path(a.out)
    if out.exists():raise ValueError('Fresh training directory')
    out.mkdir(parents=True);cfg=read('src/dunhuang/research/paired_event_r41/config.json');write(out/'protocol.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,source_manifest_sha256=sha(Path(a.data)/'manifest.json'),mode=a.mode,reconstruction_support='ENTIRE_OBSERVED_CORE_NOT_OVERWRITTEN_BY_LOCAL_CROPS',local_crop_base_frame=1));pop=Population(a.data)
    if a.mode=='checks':checks(cfg,pop,out,a.replay_from)
    else:
        gate=read(Path(a.checks)/'result.json')
        if not gate['admitted_main']:write(out/'result.json',dict(admitted=False,optimizer_steps=0,reason='PREREGISTERED_LEARNABILITY_GATE_FAILED',gate_sha256=sha(Path(a.checks)/'result.json'),operational_qualified=False));print('R41_MAIN_BLOCKED',flush=True);return
        model=initial(cfg);result=fit(model,pop,pop.schedule(cfg['main_steps'],cfg['seed']),cfg,out,'model');result.update(admitted=True,gate_sha256=sha(Path(a.checks)/'result.json'),can_route=False,operational_qualified=False);write(out/'result.json',result)
if __name__=='__main__':main()
