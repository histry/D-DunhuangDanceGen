import numpy as np,pytest
from dunhuang.research.tempo_attribution_r57.common import stats,amplification,shapley
def test_uniform_cubic_scaling_has_exact_discrete_reference():
 t=np.arange(80,dtype=float);P=np.zeros((80,22,3));P[:,:,0]=t[:,None]**3
 for speed in (.8,1.2):
  Q=P.copy();Q[:,:,0]=(t[:,None]*speed)**3
  assert stats(Q)['D3_Q95']/stats(P)['D3_Q95']==pytest.approx(speed**3,rel=1e-9)
def test_knot_stencils_are_separate_from_other_core_stencils():
 t=np.arange(30,dtype=float);P=np.zeros((30,22,3));P[:,:,0]=np.where(t<15,t,15+2*(t-15))[:,None]
 m=stats(P,[15]);assert m['knot_D3']['stencils']==3;assert m['knot_D3']['squared_mass_fraction']==pytest.approx(1)
def test_static_source_amplification_stays_unknown():
 m=stats(np.zeros((8,22,3)));assert amplification(m,m)['3']['ratio'] is None
def test_shared_support_factorial_conserves_delta():
 r=shapley({'00':3.,'01':7.,'10':9.,'11':15.});assert r['selection']+r['timing']==12;assert r['interaction']==2
def test_missing_counterfactual_is_not_interpolated():assert shapley({'00':1})['status'].startswith('NOT_CONSTRUCTIBLE')
@pytest.mark.parametrize('P',[np.zeros((3,22,3)),np.full((8,22,3),np.nan),np.zeros((8,24,3))])
def test_bad_source_observation_fails(P):
 with pytest.raises(ValueError):stats(P)
def test_teacher_preserving_score_initialization_and_gradient():
 import torch
 from dunhuang.research.tempo_attribution_r57.sdmr import preserved_scores
 base=torch.tensor([.4,.1]);raw=base.clone().requires_grad_();v=preserved_scores(raw,base);assert torch.equal(v,base);v.sum().backward();assert torch.equal(raw.grad,torch.full_like(raw,.25))
def test_time_maps_refuse_out_of_domain_and_do_not_mutate_source():
 from dunhuang.research.tempo_attribution_r57.study import Study
 from scipy.spatial.transform import Rotation
 from types import SimpleNamespace
 s=object.__new__(Study);R=np.tile(np.eye(3),(100,22,1,1));s.pop=SimpleNamespace(lib=SimpleNamespace(rotations={'x':R}),net=SimpleNamespace(joints=np.zeros((22,3)),parents=np.array([-1]+[0]*21)));s.cache={};node=dict(unit=dict(unit_uid='u',source_uid='x',source_interval=[0,100]),warp=dict(time_warp=[[0,0],[99,99]]))
 assert s.sample(node,[0,50],'linear') is None
 assert s.sample(node,[0,100])['metrics']['D3_Q95']==0
 assert np.array_equal(R,np.tile(np.eye(3),(100,22,1,1)))

def test_equal_source_queries_keep_distinct_absolute_music_maps():
 from dunhuang.research.tempo_attribution_r57.study import Study
 from types import SimpleNamespace
 s=object.__new__(Study);R=np.tile(np.eye(3),(100,22,1,1));s.pop=SimpleNamespace(lib=SimpleNamespace(rotations={'x':R}),net=SimpleNamespace(joints=np.zeros((22,3)),parents=np.array([-1]+[0]*21)));s.cache={}
 a=dict(unit=dict(unit_uid='u',source_uid='x',source_interval=[0,100]),warp=dict(time_warp=[[0,0],[99,99]]));b=dict(a,warp=dict(time_warp=[[0,300],[99,399]]))
 first=s.sample(a,[0,100]);second=s.sample(b,[300,400]);assert np.array_equal(first['R'],second['R']);assert second['time_warp'][0][1]==300 and first['time_warp'][0][1]==0
