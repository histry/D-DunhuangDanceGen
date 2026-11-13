import numpy as np
import pytest
from dunhuang.research.source_contrast.solver import project,source_constraints

def config():return dict(linear_safety_fraction=.9,max_solver_iterations=200,solver_ftol=1e-10,linear_tolerance=1e-6)
def test_identity_projection_respects_both_source_gain_constraints_and_budget():
    a=np.zeros((8250,2));g=np.eye(2);target=np.array([1.,0.]);gains=np.eye(2);m=np.array([.1,.1])
    z,r=project(a,g,target,gains,m,config())
    assert r["status"]=="SOLVED_FINITE_SUBSPACE"
    assert np.all(gains@z>=m-1e-6) and z@z<=1+1e-6 and z[1]>=.1-1e-6
def test_unresolvable_source_difference_abstains_without_global_infeasibility_claim():
    z,r=project(np.zeros((8250,2)),np.eye(2),np.ones(2),np.zeros((2,2)),np.ones(2)*.001,config())
    assert r["status"]=="UNKNOWN_SOLVER_OR_LINEAR_FEASIBILITY" and np.array_equal(z,[0,0]) and not r["global_infeasibility_claimed"]
def test_affine_source_constraints_have_correct_sign_and_jacobian():
    gains=np.array([[2.,-1.],[.5,3.]]);z=np.array([.1,.2]);m=np.array([.001,.001]);v,j=source_constraints(gains,z,m)
    h=1e-6;fd=np.column_stack([(source_constraints(gains,z+np.eye(2)[i]*h,m)[0]-source_constraints(gains,z-np.eye(2)[i]*h,m)[0])/(2*h) for i in range(2)])
    assert np.allclose(fd,j) and np.allclose(v,gains@z-m)
def test_paired_source_errors_use_visible_support_only():
    torch=pytest.importorskip("torch")
    from types import SimpleNamespace
    from dunhuang.research.source_expression.losses import reference_data
    from dunhuang.research.source_contrast.losses import errors
    raw=np.tile(np.array([1,0,0,0,1,0],np.float32),(150,8,1));mask=np.ones((150,8),bool);mask[25:40]=False
    data=reference_data((raw,mask,mask,mask),1,device="cpu")
    pred=torch.tensor(np.tile(np.r_[np.zeros(7),np.tile([1.,0,0,0,1,0],24)],(1,150,1)),dtype=torch.float32,requires_grad=True)
    rt=SimpleNamespace(bias=np.zeros(151,np.float32),scale=np.ones(151,np.float32))
    e=errors(rt,pred,data,data);assert float(e[1])==0
    grad=torch.autograd.grad(e[0],pred)[0]
    assert torch.isfinite(grad).all() and not grad[:,25:40,79:127].any()
