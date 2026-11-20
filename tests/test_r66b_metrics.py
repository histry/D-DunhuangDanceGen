"""R66-B TEST_FIXTURE: deterministic resampling/kinematic contracts, not quality."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dunhuang.research.rhythm_r66 import metrics as m


@pytest.fixture
def case():
    t=np.arange(300);v=np.zeros((300,22,3))
    v[:,:,0]=.5*np.sin(t[:,None]/25+np.arange(22)[None]/10)
    v[:,:,1]=.2*np.cos(t[:,None]/17)
    R=Rotation.from_rotvec(v.reshape(-1,3)).as_matrix().reshape(300,22,3,3)
    J=np.zeros((22,3));J[:,1]=np.arange(22)*.05;J[:,0]=np.sin(np.arange(22))*.05
    return R,J,np.arange(-1,21),[30,270]


def test_uniform_resampling_is_conformance_not_learned_following(case):
    ref,J,parents,interval=case;q=np.linspace(30,269,240);R=m.rotation_sample(ref,q)
    d=m.measure(R,ref,J,parents,q,interval)
    assert d['reconstruction_check']['maximum_angle_rad']<1e-12
    assert 'NOT_REPERFORMANCE' in d['reconstruction_check']['scope']
    assert d['versus_uniform']['ratios']['FK_path_length_mean_m']==pytest.approx(1)
    assert d['human_quality_pass'] is None and d['bridge_qualified'] is None
    assert d['Native24']=='NOT_EVALUABLE' and not d['can_route']


def test_nonlinear_time_changes_dynamics_while_endpoints_stay_fixed(case):
    ref,J,parents,interval=case;t=np.linspace(0,1,240)
    q=30+239*(t+.08*np.sin(2*np.pi*t)/(2*np.pi))
    q[0],q[-1]=30,269
    R=m.rotation_sample(ref,q);d=m.measure(R,ref,J,parents,q,interval,include_curves=True)
    assert d['reconstruction_check']['endpoint_maximum_angle_rad']<1e-12
    assert d['time_map']['source_per_target_rate']['maximum']>1
    assert d['time_map']['source_per_target_rate']['minimum']<1
    assert d['actual']['FK']['D3']['mean']!=d['reference_uniform']['FK']['D3']['mean']
    assert len(d['actual']['SO3']['D1']['curve'])==239
    assert len(d['actual']['SO3']['D3']['curve'])==237
    assert not d['time_map']['analytic_rate_certificate'] and not d['bridge_recomputed']


def test_peak_warp_clock_and_hold_usage_are_explicit(case):
    ref,J,parents,interval=case;q=np.linspace(30,269,240);R=m.rotation_sample(ref,q)
    event=dict(frame=100,type='speed_peak',body_region='upper',evidence_span=[99,102])
    hold=dict(frame=150,type='pose_hold',body_region='global',evidence_span=[145,155])
    bank=dict(beat=[1070.],accent=[1070.],accent_strength=np.ones(1300),alternate_onset=[1072.])
    d=m.measure(R,ref,J,parents,q,interval,source_events=[event,hold],audio_events=bank,
                audio_origin_frame=1000,optimized_event_ids=[hold])
    assert d['mapped_source_peaks']['audio_query_frames']==[1070.]
    assert d['mapped_source_peaks']['alignment']['primary_half_beat_half_accent']==1
    assert d['mapped_source_peaks']['alignment']['alternate_onset']['mean_nearest_error_frames']==2
    assert d['hold_turn_diagnostics'][0]['used_by_warp']
    assert d['hold_turn_diagnostics'][0]['target_read_support_touch_frames']==10


def test_near_static_amplitude_ratio_is_unknown(case):
    ref,J,parents,interval=case
    ref=np.broadcast_to(np.eye(3),(300,22,3,3)).copy();q=np.linspace(30,269,240)
    d=m.measure(m.rotation_sample(ref,q),ref,J,parents,q,interval)
    assert d['versus_uniform']['ratios']['FK_path_length_mean_m'] is None
    assert d['versus_uniform']['ratios']['FK_radial_excursion_mean_m'] is None


@pytest.mark.parametrize('change',['frames','reverse','nan','endpoint','outside','interval','rotation'])
def test_invalid_actual_source_map_contracts(case,change):
    ref,J,parents,interval=case;q=np.linspace(30,269,240);R=m.rotation_sample(ref,q)
    if change=='frames':R=R[:-1]
    elif change=='reverse':q=q[::-1]
    elif change=='nan':q[100]=np.nan
    elif change=='endpoint':q[0]=30.1
    elif change=='outside':q[-1]=270
    elif change=='interval':interval=[30,301]
    else:R[20,1]=0
    with pytest.raises(ValueError):m.measure(R,ref,J,parents,q,interval)


def test_source_reference_non240_duration_is_reported_separately(case):
    ref,J,parents,_=case;interval=[30,230];q=np.linspace(30,229,240)
    d=m.measure(m.rotation_sample(ref,q),ref,J,parents,q,interval)
    assert d['reference_native']['frames']==200 and d['reference_uniform']['frames']==240
    assert d['time_map']['uniform_target_per_source_rate']==pytest.approx(239/199)


def test_actual_output_peak_detector_does_not_reuse_mapped_source_peaks(case,monkeypatch):
    # TEST_FIXTURE interface spy: actual frozen detector code is not replaced in
    # production. Assert both actual/uniform matrices reach it and its returned
    # target-clock peaks, not source-clock labels, drive the diagnostic.
    ref,J,parents,interval=case;q=np.linspace(30,269,240);R=m.rotation_sample(ref,q)
    calls=[];stub=ModuleType('dunhuang.research.paired_event_r41.events')
    def profile(value):calls.append(value.copy());return {'TEST_FIXTURE_R':value}
    def phases(p,cfg):
        assert cfg=={'TEST_FIXTURE':True}
        return [dict(frame=1,type='speed_peak',body_region='upper',evidence_span=[0,3]),
                dict(frame=100,type='speed_peak',body_region='upper',evidence_span=[99,102])]
    stub.profiles=profile;stub.phases=phases
    for name in ('dunhuang','dunhuang.research','dunhuang.research.paired_event_r41'):
        if name not in sys.modules:
            package=ModuleType(name);package.__path__=[];monkeypatch.setitem(sys.modules,name,package)
    monkeypatch.setitem(sys.modules,stub.__name__,stub)
    source=[dict(frame=100,type='speed_peak',body_region='upper',evidence_span=[99,102])]
    d=m.measure(R,ref,J,parents,q,interval,phase_config={'TEST_FIXTURE':True},source_events=source,
                audio_events=dict(beat=[1100.],accent=[1100.]),audio_origin_frame=1000)
    assert len(calls)==2 and np.array_equal(calls[0],R)
    assert d['mapped_source_peaks']['audio_query_frames']==[1070.]
    actual=d['actual_output_peaks']
    assert [p['frame'] for p in actual['peaks']]==[100]
    assert actual['alignment']['beat']['mean_nearest_error_frames']==0
    assert not actual['source_peak_queries_reused'] and not actual['statistical_independence_or_truth']
