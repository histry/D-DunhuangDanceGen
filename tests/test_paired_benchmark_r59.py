import numpy as np
import pytest
from dunhuang.research.paired_benchmark_r59.metrics import starts,candidate_pool,relation,retrieval,clock_metrics
from dunhuang.research.paired_benchmark_r59.diagnostics import evidence_cost,decomposition

def row(raw='a',audio='song',role='val',interval=(0,150),dataset='AIST++',h=.5):return dict(raw_motion_sha256=raw,audio_sha256=audio,role=role,source_interval=list(interval),audio_interval=list(interval),dataset=dataset,uid=raw+audio+str(interval),music_activity=h,body_activity=h)
def test_full_record_grid_and_short_support():
 assert starts(0,360)==[0,75,150,210]
 assert starts(5,155)==[5]
 assert starts(0,149)==[]
def test_pool_is_label_independent_with_no_forced_positive():
 rows=[row(),row('b',h=.9),row('c','other',h=.5),row('d',role='train')]
 assert candidate_pool(rows,0)==[1,2]
 assert candidate_pool(rows,0,'activity_hard',1)==[2]
 assert relation(rows[0],rows[1])=='CLOCK_TARGET'
 assert relation(rows[0],row('a'))=='UNKNOWN'
 assert relation(rows[0],row('b',interval=(120,270)))=='UNKNOWN'
def test_fine_within_recording_scope_is_explicit():
 rows=[row(dataset='FineDance'),row('a',interval=(75,225),dataset='FineDance')]
 assert candidate_pool(rows,0)==[0,1]
 assert relation(rows[0],rows[0])=='CLOCK_TARGET'
def test_ties_do_not_get_uid_lexicographic_advantage():
 m=retrieval(np.zeros(10),[True]+[False]*9)
 assert m['recall_at1']==pytest.approx(.1)
 assert m['recall_at5']==pytest.approx(.5)
 assert m['mrr']==pytest.approx(sum(1/i for i in range(1,11))/10)
 assert retrieval([1,0],[False,False]) is None
 assert retrieval([2,1,1],[False,True,False])['mrr']==pytest.approx((.5+1/3)/2)
def test_unknown_cost_not_cheap_and_ratios_bidirectional():
 assert evidence_cost({'1':{'ratio':None}})['value'] is None
 assert evidence_cost({'1':{'ratio':2}})['value']==pytest.approx(evidence_cost({'1':{'ratio':.5}})['value'])
def test_clock_ties_and_no_padding():
 from dunhuang.research.paired_benchmark_r59.scoring import shift_inputs
 values=[np.arange(150)[:,None],np.arange(150)[:,None],np.ones((150,1),bool),np.ones(150,bool)]
 v=shift_inputs(values,15);assert len(v[0])==120 and v[1][0,0]==30 and v[1][-1,0]==149
 assert clock_metrics(np.zeros(5),[-15,-8,0,8,15],0)['accuracy']==.2
 with pytest.raises(ValueError):shift_inputs(values,16)
def test_motion_intervention_reverses_time_not_candidate_identity():
 from dunhuang.research.paired_benchmark_r59.scoring import diagonal
 mt=np.tile(np.arange(6)[:,None],(1,2));dt=np.stack([mt,mt*2]);query=[mt,mt,np.ones(6,bool),np.ones(6,bool)];cand=[dt,dt,np.ones((2,6),bool),np.ones((2,6),bool)]
 expected=(dt[:,::-1][:,::3]*mt[None,::3]).sum(-1).mean(-1)
 np.testing.assert_allclose(diagonal(query,cand,True),expected)
def test_decomposition_reconstructs_repeats_and_geometry():
 nodes=[dict(event_uid='u',source_uid='s',target=[0,240],heuristic=-.2,execution_cost=.3,source_normalized={'1':{'ratio':1}}),dict(event_uid='u',source_uid='s',target=[288,528],heuristic=-.1,execution_cost=.2,source_normalized={'1':{'ratio':1}})]
 d=decomposition(nodes,[dict(selected='G1',methods={'G1':{'gap_D3':20}})],np.ones(528),{'D3':10})
 assert d['terms']['unit_reuse']==pytest.approx(-.1)
 assert d['terms']['source_reuse']==pytest.approx(-.02)
 assert d['total']==pytest.approx(-.3-.05+1-.02-.1-.02)

def test_finedance_row6d_is_not_silently_transposed():
 from dunhuang.contracts import matrices
 from scipy.spatial.transform import Rotation
 R=Rotation.from_euler('z',.6).as_matrix();x=R[:2].reshape(6)
 np.testing.assert_allclose(matrices(x,row=True),R,atol=1e-7)
 assert not np.allclose(matrices(x),R)

def test_finedance_unequal_ends_use_only_observed_prefix():
 from dunhuang.research.paired_benchmark_r59.metrics import shared_support
 s=shared_support(1500,1497,1497);assert s['frames']==1497 and s['unmatched_motion_tail']==3
 s=shared_support(2191,2216,2216);assert s['frames']==2191 and s['unmatched_audio_feature_tail']==25 and not s['resampling'] and s['padding_frames']==0
 with pytest.raises(ValueError):shared_support(0,100,100)
