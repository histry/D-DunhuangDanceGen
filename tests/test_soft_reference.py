import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd
from dunhuang.generation.soft_reference import build_soft_reference, SCHEMA
from dunhuang.io import sha


def _plan(tmp_path, frames=225):
    x = np.zeros((frames, 151), np.float32)
    x[:, 7:] = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), 24)
    x[:, 7+6*12:13+6*12] = sixd(Rotation.from_euler('z', np.linspace(0, 1, frames)[:, None]).as_matrix())
    file = tmp_path/'source.npy'
    np.save(file, x)
    return {'frames': frames, 'events': [dict(
        reference={'path': str(file), 'sha256': sha(file)},
        music_interval=[0, frames], source_interval=[0, frames],
        time_warp=[[0, 0], [frames, frames]])]}


def test_geodesic_reference_is_soft_and_overlap_identical(tmp_path):
    p = _plan(tmp_path)
    v, m, phase, audit = build_soft_reference(p, 225, np.ones(151), np.zeros(151))
    assert v.shape == m.shape == (2, 150, 151)
    assert phase.shape == (2, 150, 1)
    np.testing.assert_array_equal(v[0, 75:], v[1, :75])
    np.testing.assert_array_equal(m[0, 75:], m[1, :75])
    assert m[:, :, 79:127].all() and not m[:, :, :79].any()
    assert not m[:, :, 127:].any()
    assert audit['schema'] == SCHEMA and audit['supported_fraction'] == 1.
    assert 0 < phase[0, 75, 0] < phase[1, 149, 0] < 1.


def test_unsupported_joint_is_masked_not_clipped(tmp_path):
    p = _plan(tmp_path, 150)
    scale = np.ones(151)
    bias = np.zeros(151)
    bias[79:85] = 5
    v, m, _, audit = build_soft_reference(p, 150, scale, bias)
    assert not m[:, :, 79:85].any()
    assert not v[:, :, 79:85].any()
    assert m[:, :, 85:127].all()
    assert audit['masked_unsupported_joint_frames'] == 150


def test_source_and_clock_contracts_are_enforced(tmp_path):
    p = _plan(tmp_path, 150)
    p['events'][0]['time_warp'] = [[0, 0], [150, 149]]
    with pytest.raises(ValueError, match='time map'):
        build_soft_reference(p, 150, np.ones(151), np.zeros(151))
    p = _plan(tmp_path, 150)
    p['events'][0]['reference']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Asset hash'):
        build_soft_reference(p, 150, np.ones(151), np.zeros(151))


def test_nonuniform_event_time_map_changes_source_phase(tmp_path):
    plan = _plan(tmp_path, 225)
    plan['events'][0]['time_warp'] = [[0, 0], [90, 75], [225, 225]]
    values, known, phase, report = build_soft_reference(plan, 225, np.ones(151), np.zeros(151))
    assert known[0, 75, 79:127].all()
    assert abs(float(phase[1, 0, 0]) - 90/225) < 1e-5
    np.testing.assert_array_equal(values[0, 75:], values[1, :75])
    assert report['supported_fraction'] == 1.0

def test_geodesic_training_phase_and_support(tmp_path):
    from dunhuang.generation.train import phase_target
    from dunhuang.contracts import matrices
    p=_plan(tmp_path,150)
    native=np.load(p['events'][0]['reference']['path'])
    target,phase=phase_target(native,np.ones(151),np.zeros(151),69,partial=True)
    assert target.shape==(150,151) and phase.shape==(150,1)
    assert abs(phase[75,0]-69/150)<1e-6
    assert not target[:,:79].any() and not target[:,127:].any()
    rotations=matrices(target[:,79:127].reshape(-1,8,6),row=True)
    np.testing.assert_allclose(np.linalg.det(rotations),1,atol=1e-5)

def test_soft_reference_rejects_unregistered_speed(tmp_path):
    p=_plan(tmp_path,150);p['events'][0]['time_warp']=[[0,0],[25,75],[150,150]]
    with pytest.raises(ValueError,match='rate'):
        build_soft_reference(p,150,np.ones(151),np.zeros(151))

def test_source_following_error_is_geodesic_and_not_a_quality_gate(tmp_path):
    from dunhuang.generation.diagnostics import reference_error,upperbody_activity
    p=_plan(tmp_path,150);motion=np.load(p['events'][0]['reference']['path'])
    error=reference_error(motion,p,np.ones(151),np.zeros(151))
    assert error['mean_radians']<1e-6 and error['observed_joint_frames']==1200
    assert upperbody_activity(motion)['mean_radians_per_second']>0
