"""Full-reference and hidden-gap conditions with the same explicit shared clock."""
import torch,numpy as np
from dunhuang.research.bridge_r38.source import packet
from dunhuang.research.validation_r51.condition import observed_phases
from dunhuang.research.coverage_r34.context_contract import context


def condition(rt,R,valid,ids,start,anchor,origin=0,phase=True,source_on=True,context_mode='both'):
 # feature_arrays masks only the contiguous bridge, and guards derivatives at event transitions.
 missing=np.flatnonzero(~valid.any(-1))
 if len(missing) and not np.all(np.diff(missing)==1):raise ValueError('Initial two-event contract cannot merge disjoint source gaps')
 gap=(int(missing[0]),int(missing[-1]+1)) if len(missing) else (len(R),len(R))
 B=len(anchor);p=packet(np.repeat(R[None],B,0),gap,['FROZEN_EVENT_REFERENCE']*B,np.repeat(ids[None],B,0))
 p.values['valid'] &= torch.tensor(valid,device='cuda')[None];p.values['velocity_valid'] &= p.values['valid']
 rp,qp=observed_phases(p.values);p.values['ref_phase']=rp if phase else torch.zeros_like(rp);p.values['query_phase']=qp[:,start:start+256] if phase else torch.zeros(B,256,8,device='cuda');p.values['mapped_source']=torch.arange(start,start+256,device='cuda',dtype=torch.float32)[None].repeat(B,1);p.values['control_mask']=torch.ones(B,256,device='cuda',dtype=torch.bool)
 if not source_on:p.values['valid'][:]=False;p.values['velocity_valid'][:]=False
 cp=context(anchor,[8,248],mode=context_mode,partial_body=True,window_origin=[origin+start]*B);cp.provenance.fill_(1);cp.coordinate_uid='PLAN_REFERENCE_SOFT_BODY_NOT_WORLD_ROOT_TRUTH'
 rt.c.clear_all();rt.c.set_source(p);rt.c.set_context(cp)
 return p,cp


def body_loss(pred,target,scales,mask=None):
 # Root and contact have no term; NaN in those target channels is legal.
 delta=pred[...,7:]-target[...,7:]
 if mask is None:mask=torch.ones_like(delta,dtype=torch.bool)
 safe=torch.where(mask,delta,torch.zeros_like(delta));pose=safe.square().sum()/mask.sum().clamp_min(1)
 losses=[pose]
 for order,scale in enumerate(scales,1):
  d=torch.diff(safe,n=order,dim=1);valid=mask.clone()
  for _ in range(order):valid=valid[:,1:]&valid[:,:-1]
  losses.append((d/scale).square().masked_select(valid).mean())
 return losses
