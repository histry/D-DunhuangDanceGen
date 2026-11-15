import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices
from dunhuang.research.contract_r28.representation import rotations
from dunhuang.research.contract_r28.input_contract import alignment_height,height_features,translate

def test_rotation_gauge_changes_numbers_without_changing_physical_motion():
    r=Rotation.random(96*22,random_state=28).as_matrix().reshape(96,22,3,3)
    raw=r[...,:2,:].reshape(96,132)
    changed=rotations(raw,"equivalent")
    assert not np.allclose(raw,changed)
    assert np.allclose(matrices(changed.reshape(96,22,6),row=True),r,atol=3e-7)
    assert np.allclose(rotations(changed),raw,atol=3e-7)

def test_canonical_reference_preserves_angular_velocity():
    from dunhuang.research.reference_r26.reference import features
    r=Rotation.from_rotvec(np.random.default_rng(4).normal(0,.02,(96*22,3))).as_matrix().reshape(96,22,3,3)
    raw=r[...,:2,:].reshape(96,132)
    a,_=features(rotations(raw,"equivalent"));b,_=features(rotations(raw))
    assert np.allclose(a[:,:,6:],b[:,:,6:],atol=2e-5)

def test_degenerate_or_nonfinite_reference_is_rejected():
    for x in (np.zeros((96,132)),np.full((96,132),np.nan),np.zeros((95,131))):
        with pytest.raises(ValueError):rotations(x)

def test_published_floor_alignment_uses_source_first_toe_average_or_minimum():
    assert alignment_height(np.array([.1,.11]))==pytest.approx(.105)
    assert alignment_height(np.array([.1,.13]))==pytest.approx(.1)
    with pytest.raises(ValueError):alignment_height([np.nan,0])

def test_C4_uses_strict_joint_specific_height_thresholds_not_speed():
    p=np.zeros((2,22,3),np.float32);p[:,[7,8,10,11],1]=[.12,.119,.05,.049]
    p[1,[7,8,10,11],0]=100
    assert np.array_equal(height_features(p),[[0,1,0,1],[0,1,0,1]])

def test_fixed_translation_roundtrip_preserves_rotations_and_C4():
    x=np.random.default_rng(3).normal(size=(256,139)).astype(np.float32);offset=np.array([2,.9,-4],np.float32)
    y=translate(x,offset);back=translate(y,-offset)
    assert np.allclose(x,back,atol=5e-7)
    assert np.array_equal(x[:,:4],y[:,:4]) and np.array_equal(x[:,7:],y[:,7:])


def test_unknown_C4_is_not_supervised_as_contact_truth_inside_Core():
    import torch
    from dunhuang.research.contract_r28.train import loss_terms
    pred=torch.ones(1,256,139,requires_grad=True)
    loss,_=loss_terms(pred,torch.zeros_like(pred),torch.zeros_like(pred))
    loss.backward()
    assert torch.count_nonzero(pred.grad[:,64:160,:4])==0
    assert pred.grad[:,64:160,4:7].norm()>0 and pred.grad[:,64:160,7:].norm()>0
    assert pred.grad[:,:64,:4].norm()>0
