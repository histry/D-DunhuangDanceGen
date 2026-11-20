"""R68 wiring regressions use synthetic data, never scientific quality labels."""
from importlib import import_module
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from dunhuang.research.local_timing_r68 import study as s
except ModuleNotFoundError:
    name='_r68_study_fixture';pkg=types.ModuleType(name);pkg.__path__=[str(Path(__file__).parent)]
    sys.modules[name]=pkg;s=import_module(name+'.study')


def case():
    return dict(key='TEST_FIXTURE',source_uid='source',source_interval=[0,240],target_interval=[1000,1240],
        audio_sha256='A'*64,time_warp=[[0,1000],[239,1239]],phase_inputs=[
        dict(type='upper_speed_peak',frame=119,evidence_span=[118,121],score=-100),
        dict(type='upper_speed_peak',frame=120,evidence_span=[119,122],score=100),
        dict(type='upper_speed_peak',frame=30,evidence_span=[29,32],score=1000)])


def test_identity_order_event_tie_ignores_scores_and_never_reselects_after_failure(monkeypatch):
    c=case();selected,proof=s.select_event(c,'upper_speed_peak')
    assert selected['frame']==119 and proof['guardable']==2
    calls=[]
    def rejected(warp,target,event,**kwargs):
        calls.append((event,kwargs));return dict(status='NOT_FEASIBLE',reason='TEST_REGISTERED_RATE_CAP')
    monkeypatch.setattr(s.warp,'intervene',rejected)
    planned,maps=s.plan_case(c)
    assert planned['selected_events']['A']['frame']==119
    assert planned['selected_events']['B'] is None
    assert [v[1]['delta'] for v in calls]==[-6,6,-6,6]
    assert [v[0]['frame'] if v[0] else None for v in calls]==[119,119,None,None]
    assert all(maps[k]['status']=='NOT_FEASIBLE' for k in ('A_early','A_late','B_early','B_late'))


def test_both_signs_use_fixed_baseline_event_and_compensation_masks():
    c,maps=s.plan_case(case())
    early=s.frame_masks(c,'A',maps['A_early']);late=s.frame_masks(c,'A',maps['A_late'])
    for key in early:np.testing.assert_array_equal(early[key],late[key])
    assert early['event_mask'][112:127].all() and early['event_mask'].sum()==15
    assert not np.any(early['support_mask']&early['guard_mask'])
    assert np.array_equal(early['compensation_mask'],early['support_mask']&~early['event_mask'])
    uniform=s.frame_masks(c,'A',maps['Uniform'])
    assert uniform['support_mask'].all() and not uniform['guard_mask'].any()


def test_infeasible_registered_position_is_kept_without_motion_metric_or_bridge_call(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('An infeasible position may not execute')
    monkeypatch.setattr(s,'rotation_sample',forbidden)
    monkeypatch.setattr(s.metrics,'measure',forbidden)
    monkeypatch.setattr(s.connectivity,'check',forbidden)
    planned,_=s.plan_case(case())
    panel,arrays=s.execute(planned,{'A_early':dict(status='NOT_FEASIBLE',reason='TEST_RATE_GUARD')},'A_early',{},None,None,None)
    assert panel['R'] is None and panel['q'] is None and not arrays
    assert panel['row']['status']=='NOT_FEASIBLE' and panel['row']['reason']=='TEST_RATE_GUARD'


def test_actual_internal_core_is_reassembled_and_pair_metrics_are_not_inherited(monkeypatch):
    c,maps=s.plan_case(case())
    source=np.broadcast_to(np.eye(3),(240,22,3,3)).copy()
    source[:,16]=Rotation.from_rotvec(np.c_[np.arange(240)*.002,np.zeros((240,2))]).as_matrix()
    q0=np.asarray(maps['E1W']['source_positions']);base=s.rotation_sample(source,q0)
    bridge=np.broadcast_to(np.eye(3),(48,22,3,3)).copy()
    base_full=np.concatenate((base,bridge,base))
    base_connection=dict(boundary_readset_sha256='same_four_frame_readset',bridge_R_sha256=s.array_hash(bridge),passes_original_screen=False)
    base_geometry=dict(bridge_R=bridge,full_R=base_full)
    ctx=dict(lib=SimpleNamespace(rotations={'source':source}),J=np.zeros((22,3)),parents=np.r_[-1,np.zeros(21,dtype=int)])
    calls=[]
    def check(c,R,ctx):
        calls.append(R.copy())
        return dict(base_connection),dict(bridge_R=bridge.copy(),full_R=np.concatenate((R,bridge,base)))
    monkeypatch.setattr(s.connectivity,'check',check)
    panel,arrays=s.execute(c,maps,'A_late',ctx,base,base_connection,base_geometry)
    assert len(calls)==1 and not np.array_equal(panel['R'],base)
    np.testing.assert_array_equal(panel['R'][:4],base[:4]);np.testing.assert_array_equal(panel['R'][-4:],base[-4:])
    assert panel['row']['bridge_invariance']['actual_bridge_bytes_identical'] is True
    assert panel['row']['connection']['passes_original_screen'] is False
    np.testing.assert_array_equal(arrays['G1_full_R'][:240],panel['R'])
    assert panel['row']['pair_dynamics']['frames']==528
    assert panel['row']['pair_dynamics']!=panel['row']['baseline_pair_dynamics']
    assert not panel['row']['can_route'] and panel['row']['artistic_quality']=='NOT_EVALUABLE'