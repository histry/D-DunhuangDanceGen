import numpy as np
import pytest

def test_content_centering_empty_gating_and_coordinate_invariance():
    torch=pytest.importorskip("torch");pytest.importorskip("pytorch3d")
    from dunhuang.research.condition_dependency.adapter import ContentContextAdapter
    torch.manual_seed(13);a=ContentContextAdapter(np.ones(151),np.zeros(151)).eval()
    x=torch.zeros(2,150,151);x[:,:,7:]=torch.tensor([1,0,0,0,1,0]*24)
    known=torch.zeros_like(x,dtype=torch.bool);known[:,52:60,4:]=True;known[:,90:98,4:]=True
    music=torch.randn(2,150,512);t=torch.tensor([100,999])
    torch.nn.init.normal_(a.output.weight,std=.01)
    assert torch.equal(a(x,torch.zeros_like(known),music,t),torch.zeros(2,150,512))
    shifted=x.clone();shifted[:,:,4:6]+=3
    assert torch.allclose(a(x,known,music,t),a(shifted,known,music,t),atol=1e-6)
    hidden=x.clone();hidden[~known]=100
    assert torch.equal(a(x,known,music,t),a(hidden,known,music,t))
    pert=x.clone();pert[:,90:98,4]+=.15
    response=(a(pert,known,music,t)-a(x,known,music,t)).square().sum()
    assert response>0;response.backward()
    assert all(torch.isfinite(p.grad).all() for p in a.parameters() if p.grad is not None)

def test_world_window_codec_roundtrip_preserves_actual_root_velocity():
    from dunhuang.research.condition_dependency.codec import encode,restore
    from dunhuang.contracts import MotionAdapter
    x=np.zeros((150,151),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],24);x[:,4]=np.arange(150)*.01+3
    edge,o=encode(x);y=restore(MotionAdapter.to_native(edge),o)
    assert np.allclose(x,y,atol=1e-6)
    assert np.allclose(np.diff(y[:,4:7],axis=0),np.diff(x[:,4:7],axis=0),atol=1e-6)


def test_balanced_units_preserve_identity_and_null_contract():
    torch=pytest.importorskip("torch");pytest.importorskip("pytorch3d")
    from dunhuang.research.condition_dependency.adapter import ContentContextAdapter
    from dunhuang.research.condition_dependency.balanced import BalancedContextAdapter,make
    torch.manual_seed(31);plain=ContentContextAdapter(np.ones(151),np.zeros(151)).eval()
    torch.nn.init.normal_(plain.output.weight,std=.01)
    scaled=BalancedContextAdapter(np.ones(151),np.zeros(151),np.ones(606)).eval()
    missing=scaled.load_state_dict(plain.state_dict(),strict=False)
    assert missing.missing_keys==["feature_inv_rms"] and not missing.unexpected_keys
    x=torch.zeros(2,150,151);x[:,:,7:]=torch.tensor([1,0,0,0,1,0]*24)
    k=torch.zeros_like(x,dtype=torch.bool);k[:,52:60,4:]=True;k[:,90:98,4:]=True
    music=torch.randn(2,150,512);t=torch.tensor([900,999])
    assert torch.equal(plain(x,k,music,t),scaled(x,k,music,t))
    assert type(make(dict(schema="r10_content_context_v1"),np.ones(151),np.zeros(151))) is ContentContextAdapter
    with pytest.raises(ValueError):BalancedContextAdapter(np.ones(151),np.zeros(151),np.zeros(606))
    inv=torch.ones(606);inv[306:453]=.1;scaled.feature_inv_rms.copy_(inv)
    assert torch.equal(scaled(x,torch.zeros_like(k),music,t),torch.zeros(2,150,512))
