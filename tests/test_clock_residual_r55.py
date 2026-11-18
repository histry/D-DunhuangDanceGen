import numpy as np
import pytest
import torch
from dunhuang.research.clock_residual_r55.common import diagonal,clock_scores,ranking_loss,residual_scores,shifted_inputs,partition,CFG,hit
from dunhuang.research.clock_residual_r55.evaluate import confidence

def test_shared_loss_multi_positive_and_gradient():
 raw=torch.tensor([.2,.1,-.1],requires_grad=True);base=raw.detach().clone();loss=ranking_loss(raw,base,[0,0,0],[True,True,False]);loss.backward();assert raw.grad[2]>0 and (raw.grad[:2]<0).all()
 with pytest.raises(ValueError):ranking_loss(raw,base,[0,0,0],[True]*3)

def test_diagonal_uses_order_not_mean():
 x=torch.eye(30)[None];valid=torch.ones(1,30,dtype=torch.bool);good=diagonal(x,x,valid,valid);bad=diagonal(x,x.flip(1),valid,valid);assert good.item()==1 and bad.item()==0
 assert torch.equal(x.mean(1),x.flip(1).mean(1))

def test_offset_sign_and_equal_support_no_circular_roll():
 x=torch.eye(120)[None];m=x[:,20:100];d=x[:,28:108];v=torch.ones(1,80,dtype=torch.bool);scores=clock_scores(m,d,v,v);assert CFG['offsets'][scores.argmax()]==-8
 with pytest.raises(ValueError):clock_scores(m[:,:20],d[:,:20],v[:,:20],v[:,:20])

def test_shifted_inputs_keep_real_source_support():
 x=torch.arange(90.)[None,:,None];mask=torch.ones_like(x,dtype=torch.bool);mv=torch.ones(1,90,dtype=torch.bool)
 m,d,_,_=shifted_inputs((x,x,mask,mv),15);assert m.shape==d.shape==(1,60,1) and m[0,0]==15 and d[0,0]==30 and d[0,-1]==89

def test_partition_group_disjoint_stable():
 meta={'groups':{'train':['g'+str(i) for i in range(48)],'val':['v0']}};p=partition(meta);assert len(p['optimization'])==40 and len(p['calibration'])==8 and not set(p['optimization'])&set(p['calibration']);assert p==partition(meta)

def test_confidence_abstains_without_evidence():
 r=[dict(group='g'+str(i%4),margin=.1,accuracy=0) for i in range(40)];assert confidence(r)['threshold'] is None
 for x in r:x['accuracy']=1
 assert confidence(r)['threshold']==0
 assert hit([0,0,0],[True,False,False])==pytest.approx(1/3)


def test_candidate_order_does_not_create_positive_index_shortcut():
 x=torch.nn.functional.normalize(torch.randn(3,30,9),dim=-1);v=torch.ones(3,30,dtype=torch.bool);original=diagonal(x[:1],x,v[:1],v);permutation=torch.tensor([2,0,1]);reordered=diagonal(x[:1],x[permutation],v[:1],v[permutation]);assert torch.equal(reordered,original[permutation])
