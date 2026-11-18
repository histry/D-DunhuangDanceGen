import argparse
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from dunhuang.research.bridge_r38.source import packet,SegmentedReference
from dunhuang.research.validation_r51.condition import PhaseSegmentedReference,observed_phases


def test_zero_phase_is_exact_segmented_checkpoint_compatibility():
    torch.manual_seed(51);torch.set_num_threads(1)
    old=SegmentedReference(4.);new=PhaseSegmentedReference(4.)
    for layer in old.out:
        torch.nn.init.normal_(layer.weight,std=.01)
    new.load_state_dict(old.state_dict(),strict=True)
    R=np.broadcast_to(np.eye(3),(1,256,22,3,3)).copy();p=packet(R,(104,152),['test'],device='cpu')
    old.set_reference(**p.values);new.set_reference(**p.values);h=torch.randn(1,256,512)
    assert torch.equal(old.inject(0,h),new.inject(0,h))
    p.values['control_mask'][:,:8]=False;new.set_reference(**p.values)
    assert torch.equal(new.inject(0,h)[:,:8],h[:,:8])


def test_phase_construction_ignores_hidden_pose_and_invalid_derivatives():
    R=np.broadcast_to(np.eye(3),(1,256,22,3,3)).copy();p=packet(R,(104,152),['test'],device='cpu')
    rp,qp=observed_phases(p.values)
    p.values['features'][:,104:152]=float('nan')
    p.values['features'][:,:, :,6:][~p.values['velocity_valid']]=float('nan')
    r2,q2=observed_phases(p.values)
    assert torch.equal(rp,r2) and torch.equal(qp,q2)
    assert not rp[:,104:152].any()
    assert torch.isfinite(qp).all()
