import numpy as np
import pytest
import torch
from dunhuang.research.local_r23.representation import basis,encode,decode,lift
from dunhuang.research.local_r23.guidance import correlation,FixedGuidance
from dunhuang.research.local_r23.external import positions24

def motion():
    x=np.zeros((225,151),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],24)
    t=np.arange(225)/224.;x[:,4]=.15*t**3;x[:,5]=1
    return x

@pytest.mark.parametrize("context",[None,(8,8),(24,16)])
def test_cubic_motion_roundtrip_and_full24(context):
    x=motion();s=encode(x,48,context);y=decode(s)
    assert y.shape==(225,151) and y.dtype==np.float32
    assert np.max(abs(y[:,4:]-x[:,4:]))<2e-6
    assert s["coefficients"].size<225*147
    assert np.array_equal(y[:,:4],np.zeros((225,4)))

def test_boundary_conditions_are_in_decoder_not_post_decoding_paste():
    x=motion();s=encode(x,48,(8,8));s["coefficients"]+=1.5
    y=decode(s);known=np.r_[0:8,217:225]
    assert np.array_equal(y[known,4:],x[known,4:])
    assert abs(y[112,4]-x[112,4])>.1
    # Unknown source interiors are not stored in the decoder's boundary state.
    assert s["boundary"].shape==(16,147)
    a=decode(s,np.array([7.,7.001,216.999,217.]))
    assert np.isfinite(a).all()

def test_endpoint_null_basis_has_zero_first_and_second_derivatives():
    from scipy.interpolate import BSpline
    count=48;knots=np.r_[np.zeros(4),np.linspace(0,1,count-2)[1:-1],np.ones(4)]
    spline=BSpline(knots,np.eye(count),3)
    for order in [0,1,2]:
        assert np.max(abs(spline(np.array([0.,1.]),nu=order)[:,3:-3]))<1e-9

def test_representation_invalid_rotations_and_time_rejected():
    x=motion();x[30,7:13]=0
    with pytest.raises(ValueError):encode(x)
    s=encode(motion())
    with pytest.raises(ValueError):decode(s,[-1])
    with pytest.raises(ValueError):basis([1.1])

def test_cfg_raw_error_identity_excludes_unknown_contacts():
    y=torch.zeros(1,225,151);k=torch.zeros_like(y,dtype=torch.bool)
    torch.manual_seed(3);u=torch.randn_like(y);c=torch.randn_like(y)
    r=correlation(u,c,y,k)
    assert r["identity_absolute_error"]<1e-12 and r["elements"]==225*147
    u[:,:,:4]=10000
    assert correlation(u,c,y,k)==r

def test_cfg1_is_conditional_and_cfg2_reproduces_original():
    class T:enabled=False
    class Old:
        adapter=None;telemetry=T()
        def __call__(self,*a):
            u=torch.ones(1,225,151)*.1;c=torch.ones_like(u)*.4
            self.branches=[u,c];g=(u+2*(c-u)).clamp(-1,1);return g,g
    x=torch.zeros(1,225,151)
    assert torch.allclose(FixedGuidance(Old(),1,None,None,[])(x,None,None,None,None,torch.tensor([2]))[0],torch.full_like(x,.4))
    assert torch.allclose(FixedGuidance(Old(),2,None,None,[])(x,None,None,None,None,torch.tensor([2]))[0],torch.full_like(x,.7))
    with pytest.raises(ValueError):FixedGuidance(Old(),1.5,None,None,[])

def test_leaf_position_does_not_invent_missing_rotations():
    parents=[-1]+[0]*23;offsets=np.ones((24,3));R=np.tile(np.eye(3),(4,22,1,1));root=np.zeros((4,3))
    p=positions24(root,R,parents,offsets)
    assert p.shape==(4,24,3) and np.array_equal(p[:,22],np.ones((4,3)))
    parents[2]=22
    with pytest.raises(ValueError):positions24(root,R,parents,offsets)

def test_modules_import_without_launching_experiments():
    import importlib
    for name in ["sample","gradients","represent","evaluate","physical_tracking","external"]:
        importlib.import_module("dunhuang.research.local_r23."+name)
