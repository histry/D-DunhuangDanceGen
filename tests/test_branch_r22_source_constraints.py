import numpy as np
from dunhuang.research.branch_r22.source_constraints import root_bounds

LIMITS=dict(root_horizontal_net_displacement_m=3.,root_horizontal_drift_speed_mps=.12,root_vertical_speed_mps_max=4.)

def test_endpoint_proof_is_invariant_to_arbitrary_editable_motion():
    x=np.zeros((225,151));x[:,4]=np.linspace(0,1,225)
    h=np.zeros_like(x,dtype=bool);h[:8,4:]=True;h[-8:,4:]=True
    a=root_bounds(x,h,LIMITS);x[8:-8,4:7]=np.random.default_rng(7).normal(size=(209,3))*10
    b=root_bounds(x,h,LIMITS)
    assert a==b and a["unavoidable_failures"]==["root_horizontal_drift_speed_mps"]
    assert abs(a["net_drift_speed_mps"]-30/224)<1e-12

def test_unknown_endpoint_does_not_fabricate_exact_drift_bound():
    x=np.zeros((225,151));h=np.zeros_like(x,dtype=bool);h[:8,4:]=True
    r=root_bounds(x,h,LIMITS)
    assert not r["endpoint_exact"] and r["net_drift_speed_mps"] is None
    assert r["complete_feasibility"]=="UNKNOWN"

def test_fixed_vertical_speed_is_only_a_lower_bound():
    x=np.zeros((225,151));x[1,5]=1.;h=np.zeros_like(x,dtype=bool);h[:8,4:]=True
    r=root_bounds(x,h,LIMITS)
    assert r["fixed_vertical_speed_lower_bound_mps"]==30. and "root_vertical_speed_mps_max" in r["unavoidable_failures"]
