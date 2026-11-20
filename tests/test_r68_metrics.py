"""Full-stencil and exact-permission regression for R68 local timing attribution."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

try:
    from dunhuang.research.local_timing_r68 import metrics as m
except ModuleNotFoundError as error:
    if not str(error.name).startswith('dunhuang.research.local_timing_r68'):
        raise
    spec=importlib.util.spec_from_file_location('r68_metrics_stage',Path(__file__).with_name('metrics.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(static=False):
    q0=np.arange(240,dtype=float);q=q0.copy()
    edit=np.zeros(240,bool);edit[80:161]=True
    event=np.zeros(240,bool);event[110:115]=True
    if not static:
        t=np.arange(300,dtype=float)
        rv=np.zeros((300,22,3));rv[:,1,2]=.2*np.sin(t/17)+.001*t
        rv[:,5,0]=.08*np.cos(t/25)
        source=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(300,22,3,3)
        z=(q0[80:161]-120)/40
        q[80:161]+=.4*(1-z*z)**4
    else:
        source=np.broadcast_to(np.eye(3),(300,22,3,3)).copy()
    baseline=m.original.rotation_sample(source,q0)
    actual=m.original.rotation_sample(source,q)
    J=np.zeros((22,3));J[:,1]=np.arange(22)*.035;J[:,0]=np.arange(22)%3*.02
    parents=np.arange(-1,21)
    return actual,baseline,J,parents,q,q0,event,edit


def test_whole_derivatives_reuse_original_operator_and_same_frame_clock():
    args=fixture();result=m.measure(*args)
    P=m.original._fk(args[0],args[2],args[3])
    expected=m.original._motion(args[0],P,True)
    assert result['whole']['actual']==expected
    for n in (1,2,3):
        record=result['stencils']['D'+str(n)]
        actual=np.array(result['whole']['actual']['FK']['D'+str(n)]['curve'])
        ref=np.array(result['whole']['baseline']['FK']['D'+str(n)]['curve'])
        assert len(actual)==240-n
        assert np.array_equal(record['FK_signed_magnitude_delta_curve'],actual-ref)
        assert record['full_stencil_count']==240-n
        assert record['partition_total']==240-n
    json.dumps(result,allow_nan=False)


def test_cross_boundary_jerk_stencils_are_not_lost_by_region_slicing():
    args=fixture();result=m.measure(*args)
    changed=m._frame_byte_changed(args[4],args[5])|m._frame_byte_changed(args[0],args[1])
    expected=np.array([changed[j:j+4].any() and not changed[j:j+4].all() for j in range(237)])
    record=result['stencils']['D3']['groups']
    assert record['cross_edited_boundary']['right_endpoint_frames']==(np.flatnonzero(expected)+3).tolist()
    assert expected.sum()>0
    # An event has5 frames but touches8 third-difference templates; taking
    # derivatives of an isolated5-frame crop would retain only2 templates.
    assert record['touch_target']['stencil_count']==8
    counts=[record['partition_'+name]['stencil_count'] for name in ('target_event','compensation','protected','untouched')]
    assert sum(counts)==237
    assert result['regions']['target_event']['frame_count']==5
    assert result['regions']['compensation']['frame_count']==76
    residual=result['stencils']['D3']['FK_derivative_residual_RMS_curve']
    for endpoint in record['fully_outside_edit_domain']['right_endpoint_frames']:
        assert residual[endpoint-3]==0


def test_clock_guard_and_body_guard_are_both_exact_not_tolerant():
    args=list(fixture());args[4]=args[4].copy();args[4][1]+=1e-12
    with pytest.raises(ValueError,match='escaped|Protected'):
        m.measure(*args)
    args=list(fixture());args[0]=args[0].copy();args[0][2,2,0,1]=-0.
    assert args[1][2,2,0,1]==0
    with pytest.raises(ValueError,match='escaped|Protected'):
        m.measure(*args)


def test_signed_zero_byte_edits_are_separate_from_numerical_edits():
    args=list(fixture(static=True));args[0]=args[0].copy()
    args[0][120,2,0,1]=-0.
    result=m.measure(*args)
    assert result['footprint']['body_byte_changed_frames']==1
    assert result['footprint']['body_numerical_changed_frames']==0
    assert result['footprint']['actual_union_changed_frames']==1
    assert result['footprint']['protected_q_and_R_bytes_exact'] is True


def test_actual_edit_cannot_escape_declared_domain_or_compensation_support():
    args=list(fixture());args[7]=args[7].copy();args[7][100]=False
    with pytest.raises(ValueError,match='escaped'):
        m.measure(*args)
    args=list(fixture());support=args[7].copy();support[120]=False
    with pytest.raises(ValueError,match='inside support'):
        m.measure(*args,support_mask=support)


@pytest.mark.parametrize('problem',['mask_float','mask_nan','q_nan','q_reverse','invalid_rotation','bad_skeleton'])
def test_invalid_observation_or_mask_is_rejected(problem):
    args=list(fixture())
    if problem=='mask_float':args[6]=args[6].astype(float)
    elif problem=='mask_nan':args[7]=args[7].astype(float);args[7][2]=np.nan
    elif problem=='q_nan':args[4]=args[4].copy();args[4][100]=np.nan
    elif problem=='q_reverse':args[4]=args[4].copy();args[4][100]=args[4][99]
    elif problem=='invalid_rotation':args[0]=args[0].copy();args[0][100,0]*=2
    else:args[3]=args[3].copy();args[3][3]=4
    with pytest.raises(ValueError):m.measure(*args)


def test_empty_and_static_regions_have_unknown_ratios_not_zero_quality():
    args=list(fixture(static=True));args[6]=np.zeros(240,bool);args[7]=np.zeros(240,bool)
    result=m.measure(*args,include_curves=False)
    assert result['geometry_retention']['ratios']['FK_path_length_mean_m'] is None
    oscillation=result['activity_normalized_oscillation']
    assert oscillation['FK_mean_speed_ratio'] is None
    assert oscillation['actual']['FK_velocity_6Hz_to_Nyquist_power_fraction'] is None
    assert result['stencils']['D3']['groups']['touch_target']['spaces']['FK']['actual']['mean'] is None
    assert 'curve' not in result['whole']['actual']['FK']['D3']
    assert 'right_endpoint_frames' not in result['stencils']['D3']['groups']['touch_edit']
    assert result['quality_pass'] is None and result['can_route'] is False
    json.dumps(result,allow_nan=False)


def test_unprotected_uniform_control_requires_explicit_zero_guard():
    args=list(fixture());args[7]=np.ones(240,bool)
    with pytest.raises(ValueError,match='protected edges'):
        m.measure(*args)
    result=m.measure(*args,guard_frames=0)
    assert result['boundary_contract']=='EXPLICIT_UNPROTECTED_CONTROL'
    assert result['regions']['protected']['frame_count']==0


def test_region_partition_retains_cross_region_memberships_without_double_count():
    args=fixture();result=m.measure(*args)
    for order in (1,2,3):
        groups=result['stencils']['D'+str(order)]['groups']
        assert sum(groups['partition_'+k]['stencil_count'] for k in ('target_event','compensation','protected','untouched'))==240-order
        assert groups['cross_frame_region_boundary']['stencil_count']>0
        assert groups['touch_target']['stencil_count']+groups['touch_compensation']['stencil_count']>=groups['touch_declared_edit']['stencil_count']
