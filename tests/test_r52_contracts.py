import copy
import numpy as np
import pytest
import torch
from dunhuang.research.strong_baseline_r52.contracts import validate,require_task,VERSION
from dunhuang.research.strong_baseline_r52.conditions import body_loss


def plan():
 event=dict(event_uid='e',source_uid='s',source_interval=[0,240],target_interval=[1500,1740],time_warp=[[0,1500],[240,1740]],phase_scope='KINEMATIC_PROXY_NOT_SEMANTIC_TRUTH',sdmr_prior=dict(qualified=False,consumed_by_denoiser=False),left_context=dict(kind='UNKNOWN'),right_context=dict(kind='UNKNOWN'),transition_budget=0)
 return dict(schema=VERSION,fps=30,target_interval=[1500,1740],root_truth='UNKNOWN',contact_truth='UNKNOWN',events=[event])


def test_complete_contract_rejects_truncated_clock_and_invented_truth():
 p=plan();assert validate(p) is p
 for change in ('warp','root','sdmr','budget'):
  q=copy.deepcopy(p)
  if change=='warp':q['events'][0]['time_warp'][-1][0]=239
  elif change=='root':q['root_truth']='INFERRED'
  elif change=='sdmr':q['events'][0]['sdmr_prior']['qualified']=True
  else:q['events'][0]['transition_budget']=1
  with pytest.raises(ValueError):validate(q)


def test_hidden_targets_cannot_enter_reference_and_root_has_no_loss():
 valid=np.ones((256,22),bool);loss=np.ones((256,139),bool);loss[:,:7]=False
 with pytest.raises(ValueError,match='Hidden target'):require_task('HIDDEN_GAP_INBETWEENING',valid,loss)
 valid[104:152]=False;require_task('HIDDEN_GAP_INBETWEENING',valid,loss)
 with pytest.raises(ValueError,match='full observed'):require_task('FULL_REFERENCE_RECONSTRUCTION',valid,loss)
 pred=torch.zeros(1,12,139,requires_grad=True);target=torch.zeros_like(pred);target[:,:,:7]=float('nan');target[:,:,7:]=torch.arange(12)[None,:,None]/12
 values=body_loss(pred,target,[.1,.1]);sum(values).backward()
 assert all(torch.isfinite(v) for v in values) and pred.grad[:,:,:7].eq(0).all() and pred.grad[:,:,7:].abs().sum()>0
