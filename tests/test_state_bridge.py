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


def test_independent_soft_sides_never_change_input_hard_mask_or_values():
    torch=pytest.importorskip("torch")
    from dunhuang.research.state_bridge.control import soft_control
    x=torch.arange(150*151,dtype=torch.float32).reshape(1,150,151);k=torch.zeros_like(x,dtype=torch.bool)
    k[:,52:60,4:]=True;k[:,90:98,4:]=True;wrong=x+1
    before=x.clone();mask=k.clone()
    for arm in ("A0","A1","A2","A3","A4","A5"):
        v,m=soft_control(x,k,wrong,arm)
        assert torch.equal(x,before) and torch.equal(k,mask)
        if arm=="A1":assert not m[:,90:98].any() and torch.equal(m[:,52:60],k[:,52:60])
        if arm=="A2":assert not m[:,52:60].any() and torch.equal(m[:,90:98],k[:,90:98])
        if arm=="A3":assert torch.equal(v[:,90:98],wrong[:,90:98]) and torch.equal(v[:,:90],x[:,:90])
        if arm=="A4":assert torch.equal(v[:,52:60],wrong[:,52:60]) and torch.equal(v[:,60:],x[:,60:])
        if arm=="A5":assert not m.any()

def test_side_gauge_matches_parent_and_does_not_leak_removed_left():
    torch=pytest.importorskip("torch");pytest.importorskip("pytorch3d")
    from dunhuang.research.condition_dependency.balanced import BalancedContextAdapter
    from dunhuang.research.state_bridge.control import SideDiagnosticAdapter,soft_control
    torch.manual_seed(6);inv=np.ones(606,np.float32);inv[306:453]=.2
    parent=BalancedContextAdapter(np.ones(151),np.zeros(151),inv).eval()
    torch.nn.init.normal_(parent.output.weight,std=.01)
    side=SideDiagnosticAdapter(np.ones(151),np.zeros(151),inv).eval();side.load_state_dict(parent.state_dict(),strict=True)
    x=torch.zeros(2,150,151);x[:,:,7:]=torch.tensor([1,0,0,0,1,0]*24);x[:,:,4]=torch.arange(150)*.01
    k=torch.zeros_like(x,dtype=torch.bool);k[:,52:60,4:]=True;k[:,90:98,4:]=True
    t=torch.tensor([100,999]);music=torch.randn(2,150,512);side.hard_anchor=x[:,59:60,4:6].clone()
    assert torch.equal(side.features(x,k,t)[0],parent.features(x,k,t)[0])
    assert torch.equal(side(x,k,music,t),parent(x,k,music,t))
    v,m=soft_control(x,k,x,"A2");f,_=side.features(v,m,t)
    assert not f[:,52:60,:151].any() and not f[:,52:60,151:302].any()
    hidden=v.clone();hidden[~m]=100
    assert torch.equal(side(v,m,music,t),side(hidden,m,music,t))
    v,m=soft_control(x,k,x,"A5");assert torch.equal(side(v,m,music,t),torch.zeros(2,150,512))
