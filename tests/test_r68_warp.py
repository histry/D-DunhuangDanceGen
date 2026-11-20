"""Controlled-clock tests do not supply motion/music artistic ground truth."""
from importlib import util
from pathlib import Path

import numpy as np
import pytest

try:
    from dunhuang.research.local_timing_r68 import warp as w
except ModuleNotFoundError:
    spec = util.spec_from_file_location('r68_warp_fixture', Path(__file__).with_name('warp.py'))
    w = util.module_from_spec(spec)
    spec.loader.exec_module(w)


def event(frame=120, span=None):
    return dict(frame=frame, evidence_span=[frame-1, frame+2] if span is None else span,
                type='speed_peak', body_region='upper', semantic_truth='UNKNOWN')


def test_frame_centres_and_nonzero_global_origin_are_not_half_open_endpoints():
    r = w.uniform([100,340],[1000,1240])
    assert r['status'] == 'LEGAL'
    assert r['source_positions'].shape == (240,)
    np.testing.assert_array_equal(r['source_positions'], np.arange(100,340))
    assert r['time_warp'] == [[100.,1000.],[339.,1239.]]
    assert r['source_interval'] == [100,340]
    assert r['proof']['rate_limits'] == [.75,1.33]


@pytest.mark.parametrize('delta',[-6.,6.])
def test_true_event_shift_inverse_continuous_limits_and_byte_exact_protection(delta):
    knots=[[100,1000],[180,1085],[260,1165],[339,1239]]
    e=event(220)
    b=w.baseline(knots,[1000,1240]);r=w.intervene(knots,[1000,1240],e,delta)
    assert r['status']=='LEGAL'
    assert r['event_target_frame_after']-r['event_target_frame_before']==delta
    assert r['event_target_frame_after_absolute']-r['event_target_frame_before_absolute']==delta
    # F(c)=c+delta, evaluated independently at the registered source event.
    c=np.interp(e['frame'],np.array(knots)[:,0],np.array(knots)[:,1])
    assert c+delta==r['event_target_frame_after_absolute']
    q=r['source_positions'];q0=b['source_positions'];mask=r['edited_frame_mask']
    assert q[~mask].tobytes()==q0[~mask].tobytes()
    assert q[:4].tobytes()==q0[:4].tobytes() and q[-4:].tobytes()==q0[-4:].tobytes()
    assert np.all(np.diff(q)>0)
    assert r['inverse_maximum_residual_target_frames']<1e-8
    assert .75<=r['proof']['source_to_target_rate_min']<=r['proof']['source_to_target_rate_max']<=1.33
    assert r['proof']['inverse_source_rate_min']==pytest.approx(1/r['proof']['source_to_target_rate_max'])
    assert r['proof']['inverse_source_rate_max']==pytest.approx(1/r['proof']['source_to_target_rate_min'])
    # Mean value theorem for the actual discrete inverse, no inverse rate reuse.
    assert (1/np.diff(q)).min()>=.75-1e-10 and (1/np.diff(q)).max()<=1.33+1e-10
    assert not r['fallback_used'] and not r['proof']['threshold_relaxed']
    assert not r['proof']['global_C1_claim'] and not r['proof']['global_C2_claim']
    assert r['dose']['max_abs_source_clock_shift_frames']>0
    for a,z in r['compensation_intervals']:
        assert 0<=a<=z<=240


def test_continuous_rate_extremum_rejects_when_endpoints_and_centre_are_safe():
    r=w.intervene([[0,0],[239,239]],[0,240],event(),delta=9)
    assert r['status']=='NOT_FEASIBLE'
    assert r['reason']=='CONTINUOUS_SOURCE_TO_TARGET_RATE_OUTSIDE_FROZEN_LIMITS'
    assert r['proof']['source_to_target_rate_min']<.75
    assert 'q' not in r and 'source_positions' not in r
    sites=r['proof']['pieces'][0]['derivative_extrema_baseline_target_frames']
    assert any(abs(s-(120-60/np.sqrt(7)))<1e-10 for s in sites)
    assert not r['fallback_used']


def test_existing_rate_cap_is_not_repaired_by_shrinking_requested_delta():
    r=w.intervene([[0,0],[100,133],[200,239]],[0,240],event(90),delta=6)
    assert r['status']=='NOT_FEASIBLE' and r['delta_target_frames']==6
    assert r['proof']['source_to_target_rate_max']>1.33
    assert not r['proof']['feasibility_repair'] and 'q' not in r


def test_no_event_evidence_crossing_and_edge_conflict_are_explicit_nonreplacement():
    k=[[100,1000],[339,1239]];t=[1000,1240]
    assert w.intervene(k,t,None)['status']=='NO_EVENT'
    cross=w.intervene(k,t,event(101,[99,102]))
    assert cross['reason']=='EVENT_EVIDENCE_NOT_WHOLE_INSIDE_SOURCE_CORE'
    edge=w.intervene([[0,0],[239,239]],[0,240],event(62))
    assert edge['reason']=='BUMP_SUPPORT_INVADES_PROTECTED_EDGE_FRAMES'
    for r in (cross,edge,w.intervene(k,t,None)):
        assert 'q' not in r and 'source_positions' not in r and not r['fallback_used']


def test_open_support_can_touch_last_protected_frame_without_editing_it():
    r=w.intervene([[0,0],[239,239]],[0,240],event(63),delta=-6)
    assert r['status']=='LEGAL' and r['bump_support_target_interval'][0]==3
    np.testing.assert_array_equal(r['source_positions'][:4],np.arange(4))
    assert r['edited_target_interval'][0]==4


def test_support_outside_curves_are_exact_and_clock_changes_only_within_declared_footprint():
    base=w.baseline([[0,0],[239,239]],[0,240])
    r=w.intervene([[0,0],[239,239]],[0,240],event(),delta=6)
    lo,hi=r['edited_target_interval'];mask=np.zeros(240,bool);mask[lo:hi]=True
    np.testing.assert_array_equal(mask,r['edited_frame_mask'])
    assert not np.any((r['source_positions']!=base['source_positions'])&~mask)
    assert r['target_evidence_interval'][0]>=r['baseline_target_evidence_interval'][0]
    assert r['target_focus_interval'][0]<=r['baseline_target_evidence_interval'][0]
    assert r['target_focus_interval'][1]>=r['target_evidence_interval'][1]


def test_uniform_is_explicit_and_illegal_endpoint_rate_does_not_return_a_clock():
    r=w.uniform([0,320],[0,240])
    assert r['status']=='NOT_FEASIBLE' and r['method']=='UNIFORM_EXPLICIT'
    assert 'q' not in r and not r['fallback_used']


@pytest.mark.parametrize('knots,target',[
    ([[0,0],[239,float('nan')]],[0,240]),
    ([[0,0],[120,100],[119,239]],[0,240]),
    ([[0,0],[239,240]],[0,240]),
    ([[False,0],[239,239]],[0,240]),
    ([[0,0],[239,239]],[0,239]),
    ([[0,0],[239,239]],[True,241]),
])
def test_invalid_map_and_budget_fail_before_sampling(knots,target):
    with pytest.raises(ValueError):w.baseline(knots,target)


@pytest.mark.parametrize('kwargs',[
    dict(delta=float('nan')),dict(delta=True),dict(half_width=0),
    dict(guard_frames=0),dict(guard_frames=1.5),dict(guard_frames=True),
])
def test_invalid_intervention_budget_is_not_converted_to_a_candidate(kwargs):
    with pytest.raises(ValueError):w.intervene([[0,0],[239,239]],[0,240],event(),**kwargs)


def test_bad_or_nonfinite_evidence_is_a_contract_error():
    for e in [event(120,[122,124]),event(120,[119,float('inf')]),event(120.5),{'frame':120}]:
        with pytest.raises(ValueError):w.intervene([[0,0],[239,239]],[0,240],e)

@pytest.mark.parametrize('delta',[-6,6])
def test_materialized_inverse_places_source_event_at_requested_target_time(delta):
    r=w.intervene([[200,1000],[439,1239]],[1000,1240],event(320),delta=delta)
    assert r['status']=='LEGAL'
    assert r['source_positions'][120+delta]==pytest.approx(320,abs=1e-9)
    at=np.flatnonzero(r['source_grid']==320)
    assert len(at)==1 and r['target_grid'][at[0]]==1120+delta