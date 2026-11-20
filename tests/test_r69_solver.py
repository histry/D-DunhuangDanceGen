"""R69 solver tests use synthetic proxy signals, not artistic truth."""
from fractions import Fraction
from importlib import util
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

try:
    from dunhuang.research.retiming_baselines_r69 import solver as s
except ModuleNotFoundError:
    spec=util.spec_from_file_location('r69_solver_fixture',Path(__file__).with_name('solver.py'))
    s=util.module_from_spec(spec);spec.loader.exec_module(s)


def _science(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:_science(v) for k,v in value.items() if k!='runtime_seconds'}
    if isinstance(value,list):return [_science(v) for v in value]
    return value


def _complete_cost(source,target,q):
    x=(source-source.mean())/source.std();y=(target-target.mean())/target.std()
    error=(np.interp(q,np.arange(len(x)),x)-y)**2
    weights=np.ones(240);weights[[0,-1]]=.5
    return float(error@weights)/239


def test_beat_dp_finds_exhaustive_small_domain_optimum_and_counts_all_legal_paths():
    r=s.beat_dp([10,250],[1000,1240],[130,170],[1,2],[1122,1158],band_frames=2,max_internal_nodes=2)
    assert r['status']=='LEGAL'
    candidates=[]
    for p in product(range(118,123),range(158,163)):
        y=np.array([0,*p,239],float);x=np.array([0,120,160,239],float)
        rates=np.diff(y)/np.diff(x)
        if not np.all((rates>=.75)&(rates<=1.33)):continue
        match=sum(w*(1-np.exp(-min(abs(v-np.array([122,158])))**2/18)) for v,w in zip(p,[1/3,2/3]))
        deform=float(np.sum(np.diff(x)/239*(rates-1)**2))
        candidates.append((match+.03*deform,p))
    best=min(candidates,key=lambda v:(v[0],v[1]))
    assert r['objective']['total']==pytest.approx(best[0],abs=1e-14)
    np.testing.assert_array_equal(np.array(r['time_warp'])[1:-1,1]-1000,best[1])
    assert r['counts']['legal_complete_paths']==len(candidates)==25
    assert r['source_positions'][0]==10 and r['source_positions'][-1]==249
    assert not r['fallback_used'] and r['proof']['endpoint_centres_exact']


def test_mixed_domain_contains_exact_noninteger_uniform_and_merges_integer_uniform_once():
    r=s.beat_dp([0,241],[0,240],[80],[1],[80])
    assert r['status']=='LEGAL'
    detail=r['source_node_options'][1]
    assert detail['uniform_target_exact']==[239,3]
    assert not detail['uniform_merged_with_integer']
    assert float(Fraction(239,3)) in r['target_node_options_local'][1]
    integer=s.beat_dp([0,240],[0,240],[120],[1],[120])
    d=integer['source_node_options'][1]
    assert d['uniform_merged_with_integer']
    assert d['integer_candidate_count']==d['actual_candidate_count']==31


def test_actual_181_source_frames_keep_legal_uniform_path_near_rate_boundary():
    src=np.sin(np.linspace(0,10,181));target=np.sin(np.linspace(0,10,240))
    r=s.rhythm_dtw([100,281],[1000,1240],src,target)
    assert r['status']=='LEGAL' and r['selected_uniform_path']
    assert r['counts']['legal_complete_paths']>=1
    assert r['proof']['source_to_target_rate_max']==pytest.approx(239/180)
    assert all(.75<=float(Fraction(*v))<=1.33 for v in r['proof']['source_to_target_rates_exact'])
    np.testing.assert_array_equal(r['source_positions'],np.linspace(100,280,240))
    assert r['source_node_options'][1]['uniform_target_exact'][1]>1


def test_source_180_frame_endpoint_rate_is_explicitly_infeasible_not_repaired():
    r=s.rhythm_dtw([0,180],[0,240],np.sin(np.arange(180)/20),np.sin(np.arange(240)/20))
    assert r['status']=='NOT_FEASIBLE' and r['reason']=='ENDPOINT_RATE_OUTSIDE_FROZEN_LIMITS'
    assert not r['endpoint_rate_legal'] and 'q' not in r and 'time_warp' not in r
    assert not r['fallback_used']


def test_rhythm_cost_is_exact_fixed240_coverage_and_can_improve_on_uniform():
    src=np.exp(-((np.arange(240)-110)/8)**2)
    target=np.exp(-((np.arange(240)-120)/8)**2)
    r=s.rhythm_dtw([0,240],[600,840],src,target)
    assert r['status']=='LEGAL'
    direct=_complete_cost(src,target,r['q'])
    assert r['objective']['matching']==pytest.approx(direct,abs=1e-11)
    assert r['objective']['actual240_profile_matching_check']==pytest.approx(direct,abs=1e-11)
    uniform=_complete_cost(src,target,np.arange(240))
    assert r['objective']['total']<uniform
    mapped=np.interp(110,np.array(r['time_warp'])[:,0],np.array(r['time_warp'])[:,1])-600
    assert 110<mapped<130
    assert r['config']['profile_chain_rule_scaling'] is False
    assert r['config']['external_system_reproduction'] is False
    assert not r['selected_uniform_path']


def test_full_profile_affine_normalization_is_frozen_and_target_origin_does_not_change_q():
    src=np.sin(np.linspace(0,8,239))+.2*np.cos(np.linspace(0,31,239))
    target=np.cos(np.linspace(0,8,240))+.1*np.sin(np.linspace(0,24,240))
    a=s.rhythm_dtw([100,339],[1000,1240],src,target)
    b=s.rhythm_dtw([100,339],[9000,9240],2*src+10,4*target-3)
    assert a['status']==b['status']=='LEGAL'
    np.testing.assert_allclose(a['q'],b['q'],rtol=0,atol=1e-12)
    assert a['objective']['matching']==pytest.approx(b['objective']['matching'],abs=1e-12)
    assert a['normalization']['source_mean']==pytest.approx(src.mean())
    assert 'ONCE_INDEPENDENTLY' in a['normalization']['scope']


def test_source_node_cap_uses_weights_before_target_music_and_keeps_chronology():
    nodes=[30,60,90,120,150,180];weights=[1,5,3,5,2,4]
    a=s.beat_dp([0,240],[0,240],nodes,weights,[50,100,150],max_internal_nodes=3)
    b=s.beat_dp([0,240],[0,240],nodes,weights,[30,80,200],max_internal_nodes=3)
    assert a['retained_source_events']==b['retained_source_events']
    assert [r['frame'] for r in a['retained_source_events']]==[60,120,180]
    assert a['counts']['source_events_not_retained']==3


@pytest.mark.parametrize('nodes,weights,beats',[
    ([],[],[30,90]),([60],[1],[]),([60],[0],[60]),([60],[1],[500]),
])
def test_beat_missing_evidence_returns_no_event_without_inventing_uniform(nodes,weights,beats):
    r=s.beat_dp([0,240],[0,240],nodes,weights,beats)
    assert r['status']=='NO_EVENT' and not r['fallback_used']
    assert 'q' not in r and 'time_warp' not in r


@pytest.mark.parametrize('src,target',[
    (np.zeros(240),np.sin(np.arange(240))),
    (np.sin(np.arange(240)),np.ones(240)),
])
def test_constant_rhythm_has_no_zscore_and_is_not_evaluable(src,target):
    r=s.rhythm_dtw([0,240],[0,240],src,target)
    assert r['status']=='NOT_EVALUABLE'
    assert r['reason']=='CONSTANT_SOURCE_OR_TARGET_RHYTHM_PROFILE'
    assert 'source_positions' not in r


def test_actual_source_support_all_frames_reciprocal_rates_and_no_new_end_frame():
    r=s.rhythm_dtw([400,718],[2000,2240],np.sin(np.arange(318)/20),np.sin(np.arange(240)/16))
    assert r['status']=='LEGAL'
    q=r['q'];assert len(q)==240 and q[0]==400 and q[-1]==717 and (np.diff(q)>0).all()
    assert min(r['interpolation_read_support'])==400 and max(r['interpolation_read_support'])==717
    assert r['proof']['inverse_rate_min']==pytest.approx(1/r['proof']['source_to_target_rate_max'])
    assert r['proof']['inverse_rate_max']==pytest.approx(1/r['proof']['source_to_target_rate_min'])
    assert r['time_warp'][-1]==[717.,2239.]


def test_complete_scientific_results_deterministic_after_runtime_is_removed():
    args=([0,240],[100,340],[80,120,160],[1,2,1],[170,220,280])
    a=s.beat_dp(*args);b=s.beat_dp(*args)
    assert json.dumps(_science(a),sort_keys=True,allow_nan=False)==json.dumps(_science(b),sort_keys=True,allow_nan=False)
    assert a['runtime_seconds']>=0 and b['runtime_seconds']>=0
    assert 'REMOVE_TO_SEPARATE_RUNTIME_TABLE' in a['runtime_scope']


@pytest.mark.parametrize('call',[
    lambda:s.beat_dp([0,240],[0,240],[120],[float('nan')],[120]),
    lambda:s.beat_dp([0,240],[0,240],[120,120],[1,1],[120]),
    lambda:s.beat_dp([0,240],[0,240],[240],[1],[120]),
    lambda:s.beat_dp([0,240],[0,240],[120],[1],[120],sigma_frames=0),
    lambda:s.beat_dp([0,240],[0,240],[120],[1],[120],band_frames=True),
    lambda:s.rhythm_dtw([0,240],[0,240],np.ones(239),np.ones(240)),
    lambda:s.rhythm_dtw([0,240],[0,240],np.r_[np.nan,np.zeros(239)],np.ones(240)),
    lambda:s.rhythm_dtw([0,240],[0,239],np.ones(240),np.ones(239)),
    lambda:s.rhythm_dtw([0,240],[0,240],np.ones(240),np.ones(240),max_internal_nodes=-1),
])
def test_malformed_signal_budget_and_original_support_raise_instead_of_silent_repair(call):
    with pytest.raises(ValueError):call()
