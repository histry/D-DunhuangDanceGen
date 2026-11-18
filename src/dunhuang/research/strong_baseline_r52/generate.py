"""Joint global-clock sampling through one frozen-length Lodge decoder."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .common import ROOT,read,write,fresh,relative_fk,digest
from .contracts import validate
from .conditions import condition
from .train import platform
from dunhuang.research.state_r27.trace import row_noise
from dunhuang.research.transition_r37.evaluate import decode
from dunhuang.research.validation_r51.common import regions


def starts(n):
 if n<256:raise ValueError('Full real 256-frame model support required')
 return sorted(set(list(range(0,n-255,128))+[n-256]))


@torch.no_grad()
def sample(rt,data,plan,phase=True,source_on=True,context_mode='both',wrong_source=False,music=None):
 T=len(data['anchor_R']);seeds=[52051,52052];B=2;origins=starts(T);anchor=torch.zeros(B,T,139,device='cuda');anchor[:,:,7:]=torch.tensor(data['anchor_R'][...,:2,:].reshape(T,132),device='cuda',dtype=torch.float32)[None]
 source=data['reference_R'].copy();valid=data['reference_valid'].copy();ids=data['event_ids'].copy()
 if wrong_source:
  good=np.flatnonzero(valid.any(-1));source[good]=data['reference_R'][np.roll(good,len(good)//2)]
  assert not np.array_equal(source[good],data['reference_R'][good]),'Wrong source must actually differ'
 m=torch.tensor(data['music35'] if music is None else music,device='cuda',dtype=torch.float32);weight=torch.hann_window(258,periodic=False,device='cuda')[1:-1].clamp_min(.01)[None,:,None]
 # All global frames are generated; only sixteen declared outer observations are copied.
 protected=torch.zeros(T,dtype=torch.bool,device='cuda');protected[:8]=True;protected[-8:]=True
 torch.manual_seed(52051);torch.cuda.manual_seed_all(52051);times=list(reversed(torch.linspace(-1,119,51).int().tolist()));cfg=np.clip(np.linspace(0,rt.d.guidance_weight*2,50),None,rt.d.guidance_weight)
 with row_noise(seeds,frames=T,device='cuda') as noise:
  x=rt.d.q_sample(anchor,torch.full((B,),119,device='cuda',dtype=torch.long),torch.randn_like(anchor))
  for t,nxt,w in zip(times[:-1],times[1:],cfg):
   total=torch.zeros_like(x);den=torch.zeros(1,T,1,device='cuda')
   for start in origins:
    condition(rt,source,valid,ids,start,anchor[:,start:start+256],origin=plan['target_interval'][0],phase=phase,source_on=source_on,context_mode=context_mode)
    ts=torch.full((B,),t,device='cuda',dtype=torch.long);_,prediction,*_=rt.d.model_predictions(x[:,start:start+256],m[start:start+256][None].repeat(B,1,1),None,ts,weight=float(w),clip_x_start=rt.d.clip_denoised)
    total[:,start:start+256]+=prediction*weight;den[:,start:start+256]+=weight
   assert (den>0).all();proposal=total/den;raw_proposal=proposal.clone();proposal[:,protected]=anchor[:,protected];proposal[:,:,:7]=anchor[:,:,:7]
   if nxt<0:x=proposal;continue
   a=rt.d.alphas_cumprod[t];an=rt.d.alphas_cumprod[nxt];epsilon=(x-a.sqrt()*proposal)/(1-a).sqrt();sigma=((1-a/an)*(1-an)/(1-a)).clamp_min(0).sqrt();x=proposal*an.sqrt()+(1-an-sigma.square()).clamp_min(0).sqrt()*epsilon+sigma*torch.randn_like(x)
 assert noise['draws']==50;rt.c.clear_all();return rt.n.unnormalize(x.clone()).cpu().numpy(),rt.n.unnormalize(raw_proposal.clone()).cpu().numpy(),noise


def metrics(raw,data,plan):
 result=[];reference=data['anchor_R'];valid=data['reference_valid'].any(-1);P0=relative_fk(reference,data['J'],data['parents']);origin=plan['target_interval'][0]
 for x in raw:
  R,_=decode(x);P=relative_fk(R,data['J'],data['parents']);theta=Rotation.from_matrix((reference.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().reshape(len(R),22)
  row=regions(R,data['J'],data['parents'],plan['regions'],origin=origin);row.update(source_core_angle_rad=float(theta[valid].mean()),source_core_fk_rmse_m=float(np.sqrt(np.mean((P[valid]-P0[valid])**2))))
  errors={}
  for order in (1,2,3):
   mask=valid.copy()
   for _ in range(order):mask=mask[1:]&mask[:-1]
   errors['D%d_reference_residual_RMS'%order]=float(np.sqrt(np.mean((np.diff(P-P0,n=order,axis=0)[mask]*30**order)**2)))
  row['source_dynamic_residual']=errors;row['root_contact']='MODEL_GAUGE_ONLY_NOT_WORLD_TRUTH';result.append(row)
 return result


def run(out,checkpoint=None,controls=False,phase=True):
 out=fresh(out);rt=platform(checkpoint);records=[];arrays={}
 for fixture in read(ROOT/'interface/result.json')['fixtures'][:1 if controls else None]:
  plan=validate(read(fixture['plan']));assert digest(plan['compiled']['path'])==plan['compiled']['sha256'];data=dict(np.load(plan['compiled']['path']));key=fixture['key'];arrays['J']=data['J'];arrays['parents']=data['parents'];arrays[key+'_Geometry']=relative_fk(data['anchor_R'],data['J'],data['parents']).astype('<f4');methods={};noises={}
  arms=['FullReference'] if not controls else ['FullReference','NoSource','ZeroPhase','NoLeft','NoRight','WrongSource','WrongMusic']
  for arm in arms:
   music=None
   if arm=='WrongMusic':
    donor=next(f for f in read(ROOT/'interface/result.json')['fixtures'] if f['key']!=key);music=np.load(read(donor['plan'])['compiled']['path'])['music35'];assert not np.array_equal(music,data['music35'])
   raw,proposal,noise=sample(rt,data,plan,phase=phase and arm!='ZeroPhase',source_on=arm!='NoSource',context_mode='no_left' if arm=='NoLeft' else 'no_right' if arm=='NoRight' else 'both',wrong_source=arm=='WrongSource',music=music)
   arrays[key+'_'+arm+'_139']=raw;arrays[key+'_'+arm+'_proposal139']=proposal;arrays[key+'_'+arm]=np.stack([relative_fk(decode(x)[0],data['J'],data['parents']) for x in raw]).astype('<f4');methods[arm]=metrics(raw,data,plan);noises[arm]=noise;print('R52_GENERATE',key,arm,flush=True)
  records.append(dict(key=key,plan=fixture['plan'],plan_sha256=fixture['plan_sha256'],frames=len(data['anchor_R']),music_interval=plan['target_interval'],methods=methods,noise=noises,window_starts=starts(len(data['anchor_R'])),protected_global_frames=16,source_observed_frames=int(data['reference_valid'].any(-1).sum()),source_visibility='COMPLETE_SELECTED_CORES; UNKNOWN_BRIDGE',target_truth='UNPAIRED_MUSIC_NO_REPERFORMANCE_TARGET',checkpoint_sha256=digest(checkpoint) if checkpoint else 'FROZEN_R51_SOURCE_CONTEXT'))
 np.savez_compressed(out/'arrays.npz',**arrays);write(out/'result.json',dict(records=records,root_contact='UNKNOWN',native24='NOT_EVALUABLE_22_OF24',whole_song_generations=0,training_steps=0,can_route=False));rt.c.zero_grad(set_to_none=True);rt.check();rt.close()
 from .view import generation
 generation(out)
