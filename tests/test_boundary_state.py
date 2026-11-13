import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd
from dunhuang.research.boundary_state.projection import project,quadratic

def motion():
    x=np.zeros((150,151),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],24)
    t=np.arange(150);x[:,4]=.001*t+.00001*t*t
    r=Rotation.from_rotvec(t[:,None]*np.array([0,.001,0])).as_matrix()
    x[:,7:13]=sixd(r)
    return x

def config(**kwargs):
    return dict(edit=[60,90],root_cap_m=2.,rotation_cap_rad=2.,chart_margin_rad=.05,root_d3_weight=16.,rotation_d3_weight=16.,**kwargs)

def test_boundary_projection_protects_outside_and_restores_discrete_derivatives():
    old=motion();raw=old.copy();raw[60:90,4]+=.3
    x,e=project(old,raw,config(),{75:[12,20]})
    assert x is not None
    assert np.array_equal(x[:60],old[:60]) and np.array_equal(x[90:],old[90:])
    assert np.array_equal(x[75,79:85],raw[75,79:85])
    for row in e["boundary_contract"]:
        assert row["root_position_residual_m"]<1e-6
        assert row["root_cross_seam_d3_max_mps3"]<.02
        assert row["angular_cross_seam_d2_max_radps3"]<.1

def test_boundary_capacity_rejection_is_necessary_not_global_infeasibility():
    old=motion();raw=old.copy();raw[60:90,4]+=1
    c=config();c["root_cap_m"]=.15
    x,e=project(old,raw,c)
    assert x is None and e["necessary_infeasibility_within_registered_caps"]
    raw[20,4]+=1
    with pytest.raises(ValueError):project(old,raw,c)

def test_quadratic_equality_solution_satisfies_free_first_order_conditions():
    y=np.arange(24,dtype=float)[:,None]**2;fixed=np.zeros(24,bool);fixed[:3]=fixed[-3:]=True
    values=y.copy();values[:3]+=2
    x=quadratic(y,fixed,values,16);d=np.diff(np.eye(24),n=3,axis=0)
    assert np.array_equal(x[fixed],values[fixed])
    assert abs(((np.eye(24)+16*d.T@d)@x-y)[~fixed]).max()<1e-9

def test_so3_boundary_angular_acceleration_with_noncommuting_axes():
    old=motion();r=np.eye(3);values=[]
    for t in range(150):
        values.append(r.copy())
        r=r@Rotation.from_rotvec([.003+t*.00001,.004+t*.00002,.002-t*.00001]).as_matrix()
    old[:,79:85]=sixd(np.asarray(values))
    raw=old.copy();raw[60:90,79:85]=sixd(np.asarray(values)[60:90]@Rotation.from_rotvec([.2,-.1,.1]).as_matrix())
    x,e=project(old,raw,config())
    assert x is not None
    assert max(v["angular_cross_seam_d2_max_radps3"] for v in e["boundary_contract"])<.1

def test_so3_chart_cut_is_rejected_without_clipping():
    old=motion();raw=old.copy()
    raw[75,67:73]=sixd(Rotation.from_rotvec([np.pi,0,0]).as_matrix())
    x,e=project(old,raw,config())
    assert x is None and e["status"]=="SO3_CHART_CUT_REJECTED"

def test_activity_guard_values_are_json_serializable_python_scalars():
    import json
    from dunhuang.research.boundary_state.study import activity
    value=activity(motion(),60,90)
    assert type(value) is float
    assert json.loads(json.dumps(dict(activity=value,guard=value>=0)))["guard"]
