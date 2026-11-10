import numpy as np
import pytest
import torch
from dunhuang.conditioning import normalize_constraints
from dunhuang.sampling import constrained_ddim
from test_sampling import Denoiser

def test_control_uses_identical_noise_and_ignores_targets():
    shape=(1,150,151)
    v=torch.zeros(shape)
    mask=torch.zeros(shape,dtype=torch.bool)
    mask[:,72,79:85]=True
    music=torch.ones(1,150,4800)
    def draw(value,mode):
        torch.manual_seed(737)
        return constrained_ddim(Denoiser(),music,value,mask,guidance=mode)
    a,na=draw(v,'none')
    v[mask]=.4
    replay,_=draw(v,'none')
    b,nb=draw(v,'hard')
    assert torch.equal(a,replay) and not torch.equal(a,b)
    assert na['noise_sha256']==nb['noise_sha256']

def test_one_unsupported_coordinate_rejects_neighborhood():
    class Scaler:
        scale_=torch.ones(151)
        min_=torch.zeros(151)
    class Normalizer:
        scaler=Scaler()
    values=np.zeros((1,150,151),np.float32)
    mask=np.zeros_like(values,dtype=bool)
    mask[0,66,79]=True
    values[0,66,79]=1.1
    with pytest.raises(ValueError,match='outside frozen'):
        normalize_constraints(Normalizer(),values,mask)
