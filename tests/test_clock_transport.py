import torch
import pytest
torch.set_num_threads(4)
from dunhuang.alignment.transport import ClockNullTransport,position_contrastive


def test_forbidden_edges_padding_and_marginals():
    torch.manual_seed(4);head=ClockNullTransport(width=8,iterations=150)
    x=torch.randn(2,12,8);y=x.clone()
    valid=torch.ones(2,12,dtype=torch.bool);valid[1,8:]=False
    t=torch.arange(12)/15
    r=head(x,y,valid,valid,t,t)
    assert r['marginal_residual'].max()<1e-4
    assert r['band_leak_mass'].max()==0
    assert r['plan'][1,8:].sum()==0
    garbage=x.clone();garbage[~valid]=float('nan')
    z=head(garbage,y,valid,valid,t,t)
    assert torch.equal(z['score'],r['score'])
    with pytest.raises(ValueError,match='increasing'):head(x,y,valid,valid,t.flip(0),t)


def test_null_capacity_can_reject_unrelated_observations_without_fixed_ratio():
    head=ClockNullTransport(width=12,iterations=150)
    x=torch.eye(12)[None];valid=torch.ones(1,12,dtype=torch.bool);t=torch.arange(12)/15
    good=head(x,x,valid,valid,t,t)
    bad=head(x,-x,valid,valid,t,t)
    assert good['score']>bad['score']
    assert good['coverage']>bad['coverage']
    assert good['music_null_fraction'].min()>=-1e-5


def test_position_weights_stop_gradient_and_associated_sources_are_not_negatives():
    torch.manual_seed(9);head=ClockNullTransport(width=8)
    x=torch.randn(3,12,8);v=torch.ones(3,12,dtype=torch.bool);t=torch.arange(12)/15
    r=head(x,x,v,v,t,t)
    r['plan'].retain_grad()
    loss=position_contrastive(r,['shared','shared','other'])
    loss.backward()
    assert r['plan'].grad is None
    assert head.music.weight.grad.isfinite().all()
    assert position_contrastive(r,['shared']*3)==0

def test_real_source_derivatives_precede_window_selection():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from dunhuang.contracts import sixd
    from dunhuang.alignment.clock_data import window_motion_features
    from dunhuang.planning.time_map import motion_features74
    n=240;t=np.arange(n)/30
    x=np.zeros((n,151),np.float32)
    x[:,7:]=np.tile([1,0,0,0,1,0],24)
    x[:,7:13]=sixd(Rotation.from_rotvec(np.column_stack((t*t*.02,np.zeros(n),np.zeros(n)))).as_matrix())
    complete,_=motion_features74(x)
    actual,mask=window_motion_features(x,60)
    np.testing.assert_array_equal(actual,complete[60:210])
    sliced,_=motion_features74(x[60:210])
    assert not np.allclose(actual[0,:3],sliced[0,:3])
    assert mask.shape==(150,74)
