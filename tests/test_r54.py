import numpy as np
import pytest
import torch
from dunhuang.research.residual_sdmr_r54.common import relation,permuted_labels,heuristic
from dunhuang.research.residual_sdmr_r54.learn import schedule,Study
from dunhuang.research.paired_event_r41.model import EventContent

def row(**kw):
 r=dict(role='train',raw_motion_sha256='a',audio_sha256='music',audio_frame_interval=[0,90],choreography_uid='dance');r.update(kw);return r

def test_relations_preserve_ambiguity_and_roles():
 a=row();assert relation(a,row(raw_motion_sha256='b'))=='CLOCK_POSITIVE'
 assert relation(a,row(raw_motion_sha256='b',audio_frame_interval=[20,110]))=='CLOCK_POSITIVE'
 assert relation(a,row(raw_motion_sha256='b',audio_frame_interval=[40,130]))=='UNKNOWN'
 assert relation(a,row(raw_motion_sha256='b',audio_sha256='other'))=='UNKNOWN'
 assert relation(a,row(raw_motion_sha256='b',audio_sha256='other',choreography_uid='other'))=='DIFFERENT_AUDIO_ASSOCIATION'
 assert relation(a,row(role='val',raw_motion_sha256='b'))=='UNKNOWN'
 assert relation(a,row(raw_motion_sha256='b',audio_frame_interval=[105,195]))=='CLOCK_NON_TARGET_TIME'

def test_shuffle_preserves_multiple_positives_and_changes_labels():
 p=[True,False,False,True,False]
 for seed in range(20):
  q=permuted_labels(p,seed);assert sum(q)==2 and q!=p and q==permuted_labels(p,seed)

def test_schedule_train_only_reproducible():
 tasks=[dict(role=role,group=str(g),frames=L) for role in ('train','val') for g in range(8) for L in (90,150,240)]
 a=schedule(tasks,10,6);assert a==schedule(tasks,10,6)
 assert all(tasks[i]['role']=='train' for ids in a for i in ids)
 assert all(len({tasks[i]['group'] for i in ids})==len(ids) for ids in a)

def test_residual_objective_gradients_use_both_modalities():
 torch.manual_seed(6);net=EventContent();base=EventContent();base.load_state_dict(net.state_dict());base.requires_grad_(False)
 inputs=[torch.randn(3,90,50),torch.randn(3,90,274),torch.ones(3,90,274,dtype=torch.bool),torch.ones(3,90,dtype=torch.bool)]
 def score(model):
  m,d,*_=model.content(*inputs);return (m[0,None]*d).sum(-1).mean(-1)
 raw=score(net);res=raw-score(base).detach();assert torch.equal(res,torch.zeros_like(res))
 loss=torch.nn.functional.softplus(-(.25*(res[0]-res[1:]))/.1).mean();loss.backward()
 assert net.music[0].weight.grad.norm()>0 and net.motion[0].weight.grad.norm()>0
 assert sum(p.numel() for p in net.parameters())==522632

def test_inference_cache_never_used_in_train_gradients():
 s=Study.__new__(Study);s.base=object();s.frozen_inference=True;s.token_cache={}
 with pytest.raises(ValueError,match='cannot enter training'):s.tokens(object(),[0])

def test_heuristic_symmetry_cannot_pass_activity_match():
 from dunhuang.research.residual_sdmr_r54.common import matched_bin
 # Equal absolute Heuristic scores can hide almost opposite motion activity.
 assert not matched_bin(-.49,.99,-.49,.01)
 assert matched_bin(-.125,.213,-.13,.222)
 assert not matched_bin(-.125,.251,-.13,.222)

def test_evaluation_balances_music_groups_instead_of_windows():
 from dunhuang.research.residual_sdmr_r54.evaluate import summarize
 # Three easy windows in one recording cannot outweigh a second music group.
 tasks=[dict(role='val',family='cross_music',group=g,recording=g,positive=[True,False]) for g in ('a','a','a','b')]
 scores={'Conditional':{i:np.array([1.,0.]) if i<3 else np.array([0.,1.]) for i in range(4)},'Heuristic':{i:np.array([0.,0.]) for i in range(4)}}
 r=summarize(tasks,scores)['val_cross_music'];assert r['methods']['Conditional']['recall_at1']['mean']==.5
 assert r['methods']['Heuristic']['recall_at1']['mean']==.5
 assert r['paired_differences']['Heuristic']['delta']==0
 assert r['methods']['Conditional']['recall_at1']['music_groups']==2
