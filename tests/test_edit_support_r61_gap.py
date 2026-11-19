import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.research.edit_support_r61.fixed_gap import solve,angles,source_balanced


def trajectory():
    axis=np.tile([.05,.1,.03],(22,1))
    t=np.arange(56)/55
    return Rotation.from_rotvec((t[:,None,None]*axis[None]).reshape(-1,3)).as_matrix().reshape(56,22,3,3)


def test_solver_ignores_every_hidden_value_and_does_not_use_phase_metadata():
    truth=trajectory();poison=truth.copy();poison[4:52]=np.nan
    a,ea=solve(truth);b,eb=solve(poison)
    changed=truth.copy();changed[4:52]=12345.0;c,_=solve(changed)
    assert ea==eb
    for name in ('SLERP','G1'):
        assert a[name] is not None
        assert np.array_equal(a[name],b[name]) and np.array_equal(a[name],c[name])
        assert np.array_equal(a[name][:4],truth[:4]) and np.array_equal(a[name][-4:],truth[-4:])
    assert ea['solver_hidden_read'] is False and ea['phase_labels_used'] is False


def test_slerp_follows_same_fixed_geodesic_target():
    truth=trajectory();curves,_=solve(truth)
    assert float(angles(curves['SLERP'][4:52],truth[4:52]).max())<1e-12
    assert curves['SLERP'].shape==(56,22,3,3)


def test_moving_hidden_truth_changes_error_not_solver_output():
    truth=trajectory();other=truth.copy()
    other[4:52]=Rotation.from_rotvec(np.tile([.3,0,0],(48*22,1))).as_matrix().reshape(48,22,3,3)
    a,_=solve(truth);b,_=solve(other)
    assert np.array_equal(a['G1'],b['G1'])
    assert angles(truth[4:52],a['G1'][4:52]).mean()<angles(other[4:52],a['G1'][4:52]).mean()


def test_observed_frames_and_contract_are_validated():
    with pytest.raises(ValueError):solve(trajectory()[:55])
    a=trajectory();a[2]=np.nan
    with pytest.raises(ValueError):solve(a)


def test_g1_rejection_is_retained_without_fallback(monkeypatch):
    monkeypatch.setitem(solve.__globals__,'segmented',lambda *args:(None,dict(status='REGISTERED_REJECTION')))
    curves,proof=solve(trajectory())
    assert curves['G1'] is None and curves['SLERP'] is not None
    assert proof['G1']['status']=='REGISTERED_REJECTION'
    assert proof['G1_rejection_fallback'] is False


def test_source_balancing_preserves_missing_failures_and_recording_units():
    rows=[dict(source_uid='a',delta=1.)]*20+[dict(source_uid='b',delta=3.),dict(source_uid='c',delta=None)]
    r=source_balanced(rows,'delta')
    assert r['mean']==2. and r['sources']==2 and r['tasks']==21
    assert source_balanced(rows,'other')['mean'] is None
    assert source_balanced(rows,'delta')==r
