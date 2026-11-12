import numpy as np
import pytest
from dunhuang.research.source_capacity.numerics import cosine,trusted

def test_derivative_gate_requires_two_scales_and_adjoint_consistency():
    cfg=dict(max_fd_relative_error=.1,min_fd_cosine=.99,max_adjoint_relative_error=1e-4)
    r=dict(adjoint_relative_error=1e-6,finite_differences=[dict(relative_error=.01,cosine=.999),dict(relative_error=.02,cosine=.999),dict(relative_error=.9,cosine=.8)])
    assert trusted(r,cfg)
    r["adjoint_relative_error"]=.01;assert not trusted(r,cfg)
    r["adjoint_relative_error"]=0;r["finite_differences"][1]["cosine"]=None;assert not trusted(r,cfg)
    assert cosine(np.ones(3),np.zeros(3)) is None

def test_jvp_vjp_and_finite_difference_linear_oracle():
    torch=pytest.importorskip("torch")
    from dunhuang.research.source_capacity.numerics import check_derivatives
    a=torch.tensor([[2.,1.],[-1.,3.]],dtype=torch.float64)
    zero=torch.zeros(2,dtype=torch.float64);d=torch.tensor([.3,-.2],dtype=torch.float64);target=torch.tensor([1.,2.],dtype=torch.float64)
    j,g,r=check_derivatives(lambda x:a@x,zero,d,target,[.25,.125,.0625])
    assert torch.allclose(j,a@d) and torch.allclose(g,a.T@target) and r["adjoint_relative_error"]<1e-12
    assert max(v["relative_error"] for v in r["finite_differences"])<1e-12

def test_hard_banded_dtw_reduces_to_same_frame_on_zero_band():
    torch=pytest.importorskip("torch")
    from dunhuang.research.source_capacity.paths import banded_dtw
    cost=torch.rand(3,7,7);allowed=torch.eye(7,dtype=torch.bool)[None].expand(3,-1,-1);valid=torch.ones(3,7,dtype=torch.bool)
    result=banded_dtw(cost,allowed,valid,valid)
    assert torch.allclose(result,cost.diagonal(dim1=-2,dim2=-1).sum(-1))
    valid[0,0]=False
    with pytest.raises(ValueError):banded_dtw(cost,allowed,valid,valid)

def test_local_operators_keep_clock_control_content_independent():
    torch=pytest.importorskip("torch")
    from dunhuang.research.source_capacity.paths import operators
    torch.set_num_threads(1);m=torch.eye(120,128)[None];d=m.clone();valid=torch.ones(1,120,dtype=torch.bool)
    cfg=dict(band_seconds=.5,time_weight=.2,null_cost=.5,epsilon=.05)
    a=operators(m,d,valid,valid,cfg);b=operators(m,-d,valid,valid,cfg,a["S3"][:1])
    assert torch.equal(a["S3"],b["S3"]) and a["S0"]>b["S0"] and a["S2"]>b["S2"]
    shifted=torch.roll(d,3,1);r=operators(m,shifted,valid,valid,cfg,a["S3"][:1])
    assert r["S1"]>r["S0"]

def test_existing_partial_path_matches_exact_solver_in_low_temperature_limit():
    torch=pytest.importorskip("torch")
    from dunhuang.alignment.partial_path import soft_path_cost,exact_path
    cost=np.array([[.1,.7,.8],[.9,.2,.3],[.8,.9,.1]],np.float32)
    allowed=np.ones((3,3),bool);v=torch.ones(1,3,dtype=torch.bool);null=torch.full((1,3),.5)
    value=soft_path_cost(torch.from_numpy(cost)[None],null,null,torch.from_numpy(allowed)[None],v,v,1e-5)
    assert abs(value.item()-exact_path(cost,.5,.5)["cost"])<1e-4

def test_target_dependent_oracle_cannot_admit_site_with_wrong_source_direction():
    pytest.importorskip("torch")
    from dunhuang.research.source_capacity.capacity import gate
    cfg=dict(min_direction_rad=.001,min_direction_cosine=0,min_relative_gain=1.1)
    record=dict(site="cond",case=0,timestep=999,numeric_trusted=True,direction_cosine=1.,
        controls=dict(learned=dict(toward_A=-.01,toward_B=.1,response=dict(normalized_response=.2)),
        oracle_target_dependent=dict(toward_A=1,toward_B=1)))
    r=gate([record],cfg)
    assert not r["train_directional_probe_pass"]["cond"] and not r["train_improved_sites"]["cond"]


def test_clipped_trajectory_record_is_independent_of_common_state_cfg_variables():
    from scipy.spatial.transform import Rotation
    from dunhuang.contracts import sixd
    from dunhuang.research.source_capacity.numerics import trajectory_entry
    a=np.zeros((150,151),np.float32);a[:,7:]=np.tile([1,0,0,0,1,0],(150,24));b=a.copy()
    b[:,79:127]=np.tile(sixd(Rotation.from_euler("x",.3).as_matrix()),(150,8))
    known=np.ones((150,8),bool)
    result=trajectory_entry(0,"C2",49,a,b,a,b,known)
    assert result["timestep"]==19 and abs(result["source_response"]["normalized_response"]-1)<1e-9
    assert result["A_preference"]>0 and result["B_preference"]>0
    assert "guided_response" not in result
    with pytest.raises(ValueError):trajectory_entry(0,"C2",50,a,b,a,b,known)
