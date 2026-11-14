import numpy as np
from dunhuang.research.boundary_reference.hermite import coefficients,evaluate,reference
from dunhuang.research.root_foot.capacity import solve
from dunhuang.contracts import sixd

def test_quintic_matches_six_endpoint_states_and_is_translation_equivariant():
    p=np.array([.1,.2,.3]);v=np.array([.2,-.1,0]);a=np.array([0,.1,.2]);q=np.array([1.,-.2,.4]);w=-v;b=-a;D=31/30
    c=coefficients(p,v,a,q,w,b,D)
    for u,states in [(0,(p,v,a)),(1,(q,w,b))]:
        for k,target in enumerate(states):assert np.allclose(evaluate(c,[u],k,D)[0],target,atol=1e-12)
    shifted=coefficients(p+4,v,a,q+4,w,b,D)
    assert np.allclose(evaluate(shifted,np.linspace(0,1,31),duration=D)-evaluate(c,np.linspace(0,1,31),duration=D),4)

def test_reference_has_no_access_to_unknown_values_or_contact_relabeling():
    x=np.zeros((150,151),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],24)
    y=x.copy();y[60:90,4:7]=2;z,meta=reference(x);z2,_=reference(y)
    assert np.array_equal(z[60:90,4:],z2[60:90,4:]) and meta["continuous_root_c2"]
    assert np.array_equal(z[:60],x[:60]) and np.array_equal(z[90:],x[90:])

def test_root_compensation_respects_caps_and_reports_incompatible_support():
    feet=np.zeros((90,4,3));feet[:,0,0]=np.arange(90)*.01;feet[:,1,0]=-np.arange(90)*.01
    mask=np.ones((90,4),bool);cfg=dict(root_l2=.01,root_d3=16,radius_m=.05,delta_speed_cap_mps=.75)
    x,e,active=solve(feet,mask,cfg)
    assert np.linalg.norm(x,axis=-1).max()<=.05+1e-12 and np.array_equal(x[:3],np.zeros((3,2)))
    assert e["irreducible_horizontal_velocity_squared"]>0 and e["multiple_support_edges"]==89
