import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.research.bridge_r38.geometry import observed_bridge,polynomial

def test_hidden_gap_poison_cannot_change_geometry_or_observed_body():
    t=np.arange(256)*.002;R=np.repeat(Rotation.from_rotvec(np.c_[t,t*0,t*0]).as_matrix()[:,None],22,axis=1);gap=[104,152]
    a=R.copy();a[104:152]=np.nan;b=R.copy();b[104:152]=Rotation.from_rotvec([2,0,0]).as_matrix()
    x=observed_bridge(a,gap);y=observed_bridge(b,gap)
    assert np.array_equal(x,y)
    assert np.array_equal(x[:104],R[:104]) and np.array_equal(x[152:],R[152:])
    assert np.max(abs(np.linalg.det(x)-1))<1e-10
    assert np.max(abs(x-R))<1e-10

def test_root_bridge_recovers_known_quadratic_without_gap_truth():
    t=np.arange(256)/30;y=np.c_[t,t*t,t*.3];x=y.copy();x[104:152]=np.nan
    assert np.allclose(polynomial(x,[104,152]),y,atol=1e-10)
    x[103]=np.nan
    with pytest.raises(ValueError):polynomial(x,[104,152])

def test_short_support_is_rejected():
    R=np.broadcast_to(np.eye(3),(20,22,3,3))
    with pytest.raises(ValueError):observed_bridge(R,[2,15])

def test_source_masks_keep_known_pose_but_not_cross_cut_derivative():
    from dunhuang.research.bridge_r38.source import feature_arrays,packet,SegmentedReference
    R=np.broadcast_to(np.eye(3),(256,22,3,3)).copy();R[104:152]=np.nan;ids=np.zeros(256,int);ids[200:]=1
    f,p,v,s=feature_arrays(R,[104,152],ids)
    assert p[152].all() and not v[152].any() and p[200].all() and not v[200].any()
    assert not p[104:152].any() and np.isfinite(f).all()
    pkt=packet(R[None],[104,152],['test'],ids[None],device='cpu');model=SegmentedReference();model.set_reference(**pkt.values)
    before=model.context[0].detach().clone();R[104:152]=Rotation.from_rotvec([1.,0,0]).as_matrix()
    model.set_reference(**packet(R[None],[104,152],['test'],ids[None],device='cpu').values)
    assert __import__('torch').equal(before,model.context[0])
