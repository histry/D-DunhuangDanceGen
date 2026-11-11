import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.planning.phase_guidance import mapped_neighborhoods
from dunhuang.contracts import matrices, sixd
from dunhuang.io import sha


def example(tmp_path):
    source=np.zeros((105,151),dtype=np.float32)
    identity=np.array([1.,0.,0.,0.,1.,0.],dtype=np.float32)
    source[:,7:]=np.tile(identity,24)
    channel=slice(7+6*12,13+6*12)
    angle=np.arange(len(source))*.01
    source[:,channel]=sixd(Rotation.from_rotvec(np.column_stack((angle*0,angle*0,angle))).as_matrix())
    path=tmp_path/'source.npy';np.save(path,source)
    event=dict(source_interval=[0,100],music_interval=[0,120],
        time_warp=[[0,0],[60,72],[100,120]],
        primitive=dict(source_frame=60,music_frame=72,joints=[12]),
        reference=dict(path=str(path),sha256=sha(path)))
    return source,dict(fps=30,frames=120,events=[event]),channel


def test_planned_music_to_source_interpolation(tmp_path):
    source,plan,channel=example(tmp_path)
    rows,weights=mapped_neighborhoods(plan,120,12)
    assert len(rows)==25 and weights[0]==0 and weights[12]==1 and weights[-1]==0
    assert rows[18]['frame']==78
    # At music frame 78 the true source coordinate is 65 (not 66).
    measured=Rotation.from_matrix(matrices(rows[18]['pose'][channel])).magnitude()
    assert abs(measured-.65)<1e-5
    assert abs(measured-.66)>.005
    np.testing.assert_allclose(rows[12]['pose'][channel],source[60,channel],atol=1e-7)


def test_planned_guidance_rejects_invalid_clock_and_warp(tmp_path):
    _,plan,_=example(tmp_path)
    with pytest.raises(ValueError,match='clock'):
        mapped_neighborhoods(plan,121,12)
    plan['events'][0]['time_warp'][1][1]=0
    with pytest.raises(ValueError,match='time map'):
        mapped_neighborhoods(plan,120,12)


def test_planned_guidance_preserves_source_hash(tmp_path):
    source,plan,_=example(tmp_path)
    path=plan['events'][0]['reference']['path']
    source[45,4]+=1.
    np.save(path,source)
    with pytest.raises(ValueError):
        mapped_neighborhoods(plan,120,12)
