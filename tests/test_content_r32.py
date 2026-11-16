import importlib
import numpy as np
import pytest

def test_core_extraction_keeps_all_observed_rotations():
    from dunhuang.research.content_r32.common import core_body,core_points
    x=np.arange(256*139).reshape(256,139)
    assert np.array_equal(core_body({"model_raw":x}),x[64:160,7:])
    assert np.array_equal(core_body({"model_raw":x[:96]}),x[:96,7:])
    points=np.zeros((256,22,3));assert core_points({"model_points":points}).shape==(96,22,3)

@pytest.mark.parametrize("name",["common","prepare","readout","partial","train"])
def test_cli_imports(name):
    importlib.import_module("dunhuang.research.content_r32."+name)


def test_partial_observation_rejects_fabricated_full_targets_and_unknowns():
    import torch
    from types import SimpleNamespace
    from dunhuang.research.content_r32.partial import partial_loss
    from dunhuang.research.reference_r26.geometry import fk22
    class Norm:
        def unnormalize(self,x):return x
    n=Norm();skel=SimpleNamespace(J=torch.arange(66,dtype=torch.float32).reshape(1,22,3)*.01,parents=[-1]+list(range(21)))
    x=torch.zeros(2,256,139);x[:,:,7:]=torch.tensor([1.,0,0,0,1,0]).repeat(22)
    body=x[:,64:160,7:].clone();p=fk22(x,skel)[:,64:160];local=p-p[:,:,:1]
    weights=dict(body=1.,coordinate_difference=.1,local_geometry=.2,prior_first7=.1,outside=.25)
    loss,_=partial_loss(x,body,local,x.clone(),n,skel,weights);assert float(loss)<1e-10
    with pytest.raises(ValueError):partial_loss(x,x[:,64:160],local,x,n,skel,weights)
    bad=body.clone();bad[0,0,0]=float('nan')
    with pytest.raises(ValueError):partial_loss(x,bad,local,x,n,skel,weights)
    with pytest.raises(ValueError):partial_loss(x,body,local+1,x,n,skel,weights)
    moved=x.clone();moved[:,:,4:7]=3.
    value,_=partial_loss(moved,body,local,moved.clone(),n,skel,weights)
    assert float(value)<1e-10 # global translation is absent from local observed truth
