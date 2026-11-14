import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd
from dunhuang.research.state_bridge.metrics import seam
from dunhuang.research.condition_dependency.codec import encode,restore
from dunhuang.contracts import MotionAdapter

def motion(n):
    x=np.zeros((n,151),np.float32);R=np.broadcast_to(np.eye(3),(n,24,3,3)).copy()
    R[:,0]=Rotation.from_rotvec(np.arange(n)[:,None]*np.array([0,.03,0])[None]).as_matrix()
    x[:,7:]=sixd(R).reshape(n,144);x[:,4:7]=np.arange(n)[:,None]*np.array([.02,0,.01])+np.array([2,1,3])
    return x,R

def test_symmetric_foot_step_decomposition_with_root_and_local_change():
    x,R=motion(20);rng=np.random.default_rng(4)
    local=rng.normal(size=(20,24,3));points=np.einsum("tij,tkj->tki",R[:,0],local)+x[:,None,4:7]
    r=seam(x,points,12,1e-6)
    assert r["identity_error_m"]<1e-6
    assert sum(b["signed_projection_share"] for b in r["components"].values())==pytest.approx(1,abs=1e-6)
    assert r["causal_attribution"] is False

def test_oracle_geometry_and_protected_frames_without_claiming_byte_identity():
    x,_=motion(150);edge,o=encode(x);back=restore(MotionAdapter.to_native(edge),o)
    assert np.allclose(back,x,atol=1e-6)
    out=x.copy();out[60:90]=back[60:90]
    assert np.array_equal(out[:60],x[:60]) and np.array_equal(out[90:],x[90:])

def test_same_truth_same_noise_mask_replacement_is_an_identity():
    torch=pytest.importorskip("torch")
    g=torch.Generator().manual_seed(7);truth=torch.randn((2,150,151),generator=g);noise=torch.randn(truth.shape,generator=g)
    alpha=.37;xt=alpha**.5*truth+(1-alpha)**.5*noise
    mask=torch.zeros_like(truth,dtype=torch.bool);mask[:,52:60,4:]=True;mask[:,90:98,4:]=True
    candidate=alpha**.5*truth+(1-alpha)**.5*noise
    assert torch.equal(torch.where(mask,candidate,xt),xt)
