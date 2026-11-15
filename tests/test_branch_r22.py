import importlib,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from dunhuang.research.branch_r22.objective import supervise
from dunhuang.research.branch_r22.local_contract import require_native24,common22

def fixture():
    target=torch.zeros(1,225,151);known=torch.zeros_like(target,dtype=torch.bool)
    known[:,:8,4:]=True;known[:,-8:,4:]=True
    return target,known

def test_branch_loss_resolves_guided_output_compensation():
    y,k=fixture();u=torch.full_like(y,.4,requires_grad=True);c=torch.full_like(y,.2,requires_grad=True)
    g=(2*c-u).clamp(-1,1)
    u2=u+.3;c2=c+.15;g2=(2*c2-u2).clamp(-1,1)
    assert torch.allclose(g,g2,atol=1e-7)
    a=supervise("G0",u,c,g,y,k)["unknown"];b=supervise("G0",u2,c2,g2,y,k)["unknown"]
    assert abs(float(a-b))<1e-12
    assert supervise("G1",u2,c2,g2,y,k)["unknown"]>supervise("G1",u,c,g,y,k)["unknown"]

def test_raw_branch_gradient_survives_saturation_and_both_branches_train():
    y,k=fixture();u=torch.full_like(y,2.,requires_grad=True);c=torch.full_like(y,3.,requires_grad=True);g=(2*c-u).clamp(-1,1)
    clipped=supervise("G1",u,c,g,y,k)["unknown"];raw=supervise("G2",u,c,g,y,k)["unknown"]
    gc=torch.autograd.grad(clipped,(u,c),retain_graph=True);gr=torch.autograd.grad(raw,(u,c))
    assert all(torch.count_nonzero(x)==0 for x in gc)
    assert all(torch.count_nonzero(x)>0 for x in gr)
    assert all(torch.count_nonzero(x[:,:,:4])==0 for x in gr)

def test_detached_diagnostic_branch_cannot_silently_train():
    y,k=fixture();raw=torch.ones_like(y)
    with pytest.raises(AssertionError,match="Detached"):supervise("G1",raw,raw,raw,y,k)

def test_physical_path_and_guided_prediction_are_identical_across_arms(monkeypatch):
    m=importlib.import_module("dunhuang.research.branch_r22.objective")
    y,k=fixture();u=torch.full_like(y,.1,requires_grad=True);c=torch.full_like(y,.3,requires_grad=True)
    class P:
        def __call__(self,*args):
            self.branches=[u,c];g=(2*c-u).clamp(-1,1);return g,g
    def losses(p,y,*args):
        basic=importlib.import_module("dunhuang.research.objective_r19.objective").reconstruction(p,y,k)[0]
        return dict(basic,physics=p.square().mean()),{}
    monkeypatch.setattr(m,"losses",losses);preds=[];phys=[]
    state=(y,None,None,y,k,k,None,None,y)
    for arm in ("G0","G1","G2"):
        p,t,counts,active,total=m.build({},P(),{},state,{"physics":.2},{},arm);preds.append(p);phys.append(active["physics"])
        assert total.requires_grad
    assert all(torch.equal(preds[0],p) for p in preds) and all(torch.equal(phys[0],p) for p in phys)

def test_incomplete_external_skeleton_cannot_be_padded_into_full_native():
    with pytest.raises(ValueError,match="INCOMPLETE_NATIVE24"):require_native24(np.zeros((256,139)),[True]*22+[False]*2,30)
    with pytest.raises(ValueError,match="INCOMPLETE_NATIVE24"):require_native24(np.zeros((256,151)),[True]*22+[False]*2,30)
    x=np.zeros((225,151));assert require_native24(x,[True]*24,30) is x
    m=common22(np.zeros((256,22,3)));assert m["full_native_status"]=="NOT_EVALUABLE" and not m["operational_qualified"]

@pytest.mark.parametrize("module",["engine","data","train","rollout","sanity","lodge"])
def test_registered_command_modules_import_without_running_experiments(module):
    importlib.import_module("dunhuang.research.branch_r22."+module)
