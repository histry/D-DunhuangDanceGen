import numpy as np
import pytest
from dunhuang.research.sdmr_event_rank.scorer import masked_diagonal,batch_partial,partial_score,retrieval_metrics
from dunhuang.research.sdmr_event_rank.population import relation


def test_padding_cannot_raise_score_and_real_non_grid_offsets_are_supported():
    positions=np.arange(6,24);m=np.eye(18);d=np.roll(m,2,axis=0);valid=np.ones(18,bool)
    score,coverage=masked_diagonal(m,d,valid,valid,positions,2)
    assert score==1. and coverage>0.8
    d[2:]=1000;dv=valid.copy();dv[2:]=False
    score,coverage=masked_diagonal(m,d,valid,dv,positions,2)
    assert score==-1. and coverage==0.


def test_partial_batch_is_exact_monotone_drop_solution_with_masks():
    rng=np.random.default_rng(43);m=rng.normal(size=(24,8));m/=np.linalg.norm(m,axis=-1,keepdims=True);d=m.copy();dv=np.ones(24,bool);dv[9:12]=False;mv=np.ones(24,bool);mv[:2]=False;cfg=dict(token_stride=3,path_band_frames=15,path_drop_cost=.3,path_null_penalty=.3,path_warp_penalty=.05);q=dict(mt=m,mv=mv,positions=np.arange(6,30));c=dict(dt=d,dv=dv)
    score,cov,cost=batch_partial(q,[c],cfg);s,p,path=partial_score(m[::3],d[::3],mv[::3],dv[::3],q['positions'][::3],cfg)
    assert score[0]==pytest.approx(s,abs=1e-7) and cov[0]==p and cost[0]==pytest.approx(path['cost'],abs=1e-7)
    a=np.array(path['matches']);assert (np.diff(a,axis=0)>0).all()


def test_ties_report_expected_chance_instead_of_source_id_order():
    v=retrieval_metrics(np.zeros(10),[True]+[False]*9)
    assert v['recall_at1']==pytest.approx(.1) and v['recall_at5']==pytest.approx(.5)
    assert v['mrr']==pytest.approx(sum(1/i for i in range(1,11))/10)
    assert retrieval_metrics([2.,1.,0.],[False,True,False])['mrr']==.5


def test_unknown_clock_overlap_and_same_recording_are_not_fabricated_negatives():
    cfg=dict(positive_iou=.6,negative_gap_frames=15);a=dict(role='train',raw_motion_sha256='a',audio_sha256='m',audio_frame_interval=[0,90]);b=dict(a,raw_motion_sha256='b');assert relation(a,b,cfg)=='CLOCK_POSITIVE'
    b['audio_frame_interval']=[60,150];assert relation(a,b,cfg)=='UNKNOWN'
    b['audio_frame_interval']=[105,195];assert relation(a,b,cfg)=='CLOCK_NEGATIVE'
    b['raw_motion_sha256']='a';assert relation(a,b,cfg)=='UNKNOWN'
    b['role']='val';assert relation(a,b,cfg)=='UNKNOWN'
