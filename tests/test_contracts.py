import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.contracts import MotionAdapter, matrices, sixd, Q
from dunhuang.conditioning import anchor_windows
from dunhuang.music import clock_plan

def poses(n=8):
    x=np.zeros((n,151),np.float32)
    rng=np.random.default_rng(7)
    x[:,:4]=rng.random((n,4));x[:,4:7]=rng.normal(size=(n,3))
    x[:,7:]=sixd(Rotation.random(n*24,random_state=rng).as_matrix()).reshape(n,144)
    return x

def test_roundtrip_preserves_contacts_root_rotations():
    x=poses();origin=np.array([2.,3.,4.])
    y=MotionAdapter.to_native(MotionAdapter.to_edge(x,origin),origin)
    np.testing.assert_allclose(x,y,atol=1e-6)
    np.testing.assert_array_equal(x[:,:4],y[:,:4])

def test_fk_world_coordinate_equivalence():
    x=poses();edge=MotionAdapter.to_edge(x)
    native_r=matrices(x[:,7:].reshape(-1,24,6));edge_r=matrices(edge[:,7:].reshape(-1,24,6),row=True)
    # Nontrivial chain and offsets: world conversion affects only the root rotation.
    offsets=np.random.default_rng(2).normal(size=(24,3));offsets[0]=0
    def fk(r,root):
        globals=[r[:,0]];positions=[root]
        for j in range(1,24):
            p=(j-1)//2
            positions.append(positions[p]+np.einsum('tij,j->ti',globals[p],offsets[j]))
            globals.append(globals[p]@r[:,j])
        return np.stack(positions,1)
    np.testing.assert_allclose(fk(native_r,x[:,4:7])@Q.T,fk(edge_r,edge[:,4:7]),atol=5e-6)

@pytest.mark.parametrize('frame',[0,74,75,149,150,224])
def test_anchors_share_global_clock(frame):
    v,m=anchor_windows(225,[dict(frame=frame,pose=poses(1)[0],joints=[12,18])])
    np.testing.assert_array_equal(m[0,75:],m[1,:75])
    np.testing.assert_array_equal(v[0,75:],v[1,:75])
    assert not m[:,:,:7].any()

def test_anchor_rejects_outside_clock():
    with pytest.raises(ValueError):anchor_windows(150,[dict(frame=150,pose=poses(1)[0])])

def test_degenerate_rotation_rejected():
    with pytest.raises(ValueError):MotionAdapter.to_edge(np.zeros((1,151),np.float32))

def test_music_clock_has_no_extra_window_on_exact_boundary():
    assert clock_plan(5*16000,16000)['windows']==1
    assert clock_plan(15*16000,16000)['windows']==5
    assert clock_plan(15*16000+1,16000)['windows']==6
