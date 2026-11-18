"""Independent Module II body-only adaptation on real continuous recordings."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .common import read,write,fresh,digest
from .contracts import require_task
from .conditions import condition,body_loss
from dunhuang.research.bridge_r38.study import runtime
from dunhuang.research.coverage_r34.context_train import restore_context
from dunhuang.research.validation_r51.condition import install
from dunhuang.research.transition_r37.evaluate import decode
from dunhuang.research.bridge_r38.geometry import observed_bridge
from dunhuang.research.lodge_adapter_r25.model import digest as model_digest


def platform(checkpoint=None,trainable=False):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(52052);torch.cuda.manual_seed_all(52052)
 rt=runtime(read('src/dunhuang/research/bridge_r38/config.json'));restore_context(rt,'outputs/bridge_r38_01/training/context.pt');install(rt)
 if checkpoint:
  p=torch.load(checkpoint,map_location='cpu',weights_only=True);rt.c.change.load_state_dict(p['source'],strict=True);rt.c.context_branch.load_state_dict(p['context'],strict=True)
 rt.c.change.requires_grad_(trainable);rt.c.context_branch.requires_grad_(trainable);rt.c.eval();rt.d.model.eval();rt.frozen=rt.fingerprint();return rt


def state(rt):return dict(source={k:v.detach().cpu().clone() for k,v in rt.c.change.state_dict().items()},context={k:v.detach().cpu().clone() for k,v in rt.c.context_branch.state_dict().items()})

def fixed(rt):return dict(backbone=model_digest(rt.d.model),io=model_digest(rt.io),general=model_digest(rt.c.general))

def schedule(meta,steps):
 rng=np.random.default_rng(52052);groups={d:[r for r in meta['records'] if r['domain']==d and r['role']=='train'] for d in ('AIST++','ChangE')};rows=[]
 for i in range(steps):
  chosen=[groups[d][k] for d in groups for k in rng.choice(len(groups[d]),2,replace=False)]
  rows.append(dict(step=i,keys=[r['key'] for r in chosen],task='FULL_REFERENCE_RECONSTRUCTION' if i%2==0 else 'HIDDEN_GAP_INBETWEENING',t=[119,519,59,999][(i//2)%4],seed=520520+i))
 return rows


def batch(rt,arrays,row,phase):
 targets=[];inputs=[];music=[];sources=[];valids=[]
 for key in row['keys']:
  raw=arrays[key+'_truth'];R=decode(raw)[0];valid=np.ones((256,22),bool);source=R.copy()
  if row['task']=='HIDDEN_GAP_INBETWEENING':valid[104:152]=False;source[104:152]=np.nan;R=observed_bridge(source,(104,152))
  target=raw.copy();target[:,:7]=np.nan;target_mask=np.ones((256,139),bool);target_mask[:,:7]=False;require_task(row['task'],valid,target_mask)
  targets.append(target);inputs.append(R[...,:2,:].reshape(256,132));music.append(arrays[key+'_music']);sources.append(source);valids.append(valid)
 a=torch.zeros(4,256,139,device='cuda');a[:,:,7:]=torch.tensor(np.stack(inputs),device='cuda',dtype=torch.float32);y=torch.tensor(np.stack(targets),device='cuda',dtype=torch.float32);m=torch.tensor(np.stack(music),device='cuda',dtype=torch.float32)
 # Install each source in one batched packet, preserving its own masks/segments.
 from dunhuang.research.bridge_r38.source import packet
 from dunhuang.research.validation_r51.condition import observed_phases
 from dunhuang.research.coverage_r34.context_contract import context
 gap=(104,152) if row['task']=='HIDDEN_GAP_INBETWEENING' else (256,256)
 source=packet(np.stack(sources),gap,row['keys']);rp,qp=observed_phases(source.values);source.values['ref_phase']=rp if phase else torch.zeros_like(rp);source.values['query_phase']=qp if phase else torch.zeros_like(qp)
 cp=context(a,[104,152] if row['task']=='HIDDEN_GAP_INBETWEENING' else [8,248],partial_body=True);cp.provenance.fill_(1);rt.c.clear_all();rt.c.set_source(source);rt.c.set_context(cp)
 return a,y,m


def run(out,phase=True,replay=None):
 out=fresh(out);data='outputs/source_context_r42_01/data_v2';meta=read(data+'/result.json');arrays=np.load(data+'/arrays.npz');assert digest(data+'/arrays.npz')==meta['arrays_sha256'];ledger=schedule(meta,240)
 rt=platform(trainable=True);frozen=fixed(rt);params=list(rt.c.change.parameters())+list(rt.c.context_branch.parameters());opt=None if replay else torch.optim.AdamW(params,lr=1e-4)
 train=np.stack([arrays[r['key']+'_truth'][:,7:] for r in meta['records'] if r['role']=='train']);scales=[float(max(np.sqrt(np.mean(np.diff(train,n=k,axis=1)**2)),.01)) for k in (1,2)];history=[]
 weights=[1.,.1,.1];torch.save(dict(**state(rt),step=0,phase=phase),out/'initial.pt')
 def loss(row):
  torch.manual_seed(row['seed']);torch.cuda.manual_seed_all(row['seed']);a,y,m=batch(rt,arrays,row,phase);ts=torch.full((4,),row['t'],device='cuda',dtype=torch.long);xt=rt.d.q_sample(a,ts,torch.randn_like(a));pred=rt.d.model(xt,m,None,ts,cond_drop_prob=0);mask=torch.ones_like(pred[...,7:],dtype=torch.bool);mask[:,:8]=False;mask[:,-8:]=False
  values=body_loss(pred,y,scales,mask);return values,sum(w*v for w,v in zip(weights,values))
 # Calibrate dynamic terms using only one registered Train batch and actual gradients.
 values,total=loss(ledger[0]);norms=[]
 for value in values:
  grads=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True);norms.append(float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None))))
 assert min(norms)>0 and np.isfinite(norms).all();weights=[1.]+[float(.1*norms[0]/g) for g in norms[1:]]
 if not phase:weights=read('outputs/strong_baseline_r52_01/train_main/calibration.json')['weights']
 rt.c.clear_all();write(out/'calibration.json',dict(train_keys=ledger[0]['keys'],scales=scales,gradient_norms=norms,weights=weights,selection='TRAIN_ONLY_EQUAL_GRADIENT_TENTH_RATIO'))
 check_steps=(0,119,239)
 if replay:
  old=read(str(replay)+'/result.json');assert old['calibration']==read(out/'calibration.json')
  for row in [ledger[i] for i in check_steps]:
   before=torch.load(str(replay)+'/step%d_pre.pt'%row['step'],map_location='cpu',weights_only=True);rt.c.change.load_state_dict(before['source']);rt.c.context_branch.load_state_dict(before['context']);rt.c.zero_grad(set_to_none=True);values,total=loss(row);total.backward();grads=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in params]).cpu().numpy();saved=np.load(str(replay)+'/gradient%d.npz'%row['step']);assert np.array_equal(grads,saved['gradient']);assert float(total)==old['history'][row['step']]['loss'];history.append(dict(step=row['step'],exact_gradient=True));rt.c.clear_all()
  write(out/'result.json',dict(snapshot_checks=history,optimizer_calls=0,frozen=fixed(rt)));assert fixed(rt)==frozen;rt.frozen=rt.fingerprint();rt.c.zero_grad(set_to_none=True);rt.close();return
 for row in ledger:
  step=row['step']
  if step in check_steps:torch.save(state(rt),out/f'step{step}_pre.pt')
  opt.zero_grad(set_to_none=True);values,total=loss(row);assert torch.isfinite(total);total.backward();g=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in params]);norm=float(g.norm());assert np.isfinite(norm) and norm>0
  if step in check_steps:np.savez_compressed(out/f'gradient{step}.npz',gradient=g.detach().cpu().numpy())
  torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();history.append(dict(row,loss=float(total.detach()),terms=[float(v.detach()) for v in values],gradient_norm=norm));rt.c.clear_all()
  if step in check_steps:torch.save(state(rt),out/f'step{step}_post.pt')
  if (step+1)%40==0:print('R52_TRAIN',phase,step+1,history[-1]['loss'],flush=True)
 assert fixed(rt)==frozen;torch.save(dict(**state(rt),step=240,phase=phase),out/'model.pt');write(out/'result.json',dict(phase=phase,history=history,calibration=read(out/'calibration.json'),train_keys=sorted({k for r in ledger for k in r['keys']}),trainable_parameters=sum(p.numel() for p in params),optimizer_steps=240,frozen=frozen,sdmr_gate_used=False,root_contact_loss=False,can_route=False));rt.c.zero_grad(set_to_none=True);rt.frozen=rt.fingerprint();rt.close()
