import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd,matrices
from dunhuang.research.root_pose_r15.geometry import input_mask,shuffled_target,response,qualification,protected

def config():
    return json.loads((Path(__file__).parents[1]/"src/dunhuang/research/root_pose_r15/config.json").read_text())

def motion():
    x=np.zeros((150,151),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],(150,24))
    x[:,4]=np.arange(150)*.01;x[:,5]=1
    return x

def test_input_permissions_and_dimension():
    for arm in ("J1","J2","J3","J4"):
        k=input_mask((1,150,151),arm,[1,2,4,5,7,8])
        assert int(k.sum())==(270 if arm=="J1" else 1350)
        assert not k[:,:60].any() and not k[:,90:].any() and not k[:,:,:4].any()
        assert not k[:,:,7+6*20:13+6*20].any()
    with pytest.raises(ValueError):input_mask((1,150,151),"invalid",[])

def test_shuffle_keeps_recipient_anchors_and_uses_relative_gauge():
    x=motion();y=x.copy();R=Rotation.from_euler("y",.8).as_matrix()
    y[:,4:7]=x[:,4:7]@R.T+[8,0,-3];y[:,7:13]=sixd(R@matrices(x[:,7:13]))
    z=shuffled_target(x,y)
    assert np.array_equal(z[:60],x[:60]) and np.array_equal(z[90:],x[90:]) and np.array_equal(z[:,:4],x[:,:4])
    assert np.allclose(z[60:90,4:7],x[60:90,4:7],atol=1e-6)
    assert np.allclose(matrices(z[60:90,7:13]),matrices(x[60:90,7:13]),atol=1e-6)

def test_unknown_recipient_truth_cannot_leak_into_shuffled_target():
    x=motion();y=motion();x[60:90,4:]=np.nan
    z=shuffled_target(x,y)
    assert np.isfinite(z).all()
    assert np.allclose(z[60:90,4:7],y[60:90,4:7])

def test_physical_response_radius_and_protected_frames():
    cfg=config();x=motion();y=x.copy();y[60,4]+=.03
    assert not response(y,x,cfg)["within_radius"]
    y=x.copy();y[59,4]+=1
    with pytest.raises(ValueError):response(y,x,cfg)
    y=x.copy();y[70,7+6:13+6]=sixd(Rotation.from_euler("x",.06).as_matrix())
    assert not response(y,x,cfg)["within_radius"]

def test_capacity_gate_retains_every_group_and_requires_joint_gain():
    cfg=config();rr=[]
    for i in range(8):
        for arm in cfg["arms"]:
            rr.append(dict(case=i,arm=arm,accepted=arm=="J3",protection={"both":arm=="J3"},
                metrics={"feet_rmse_m":.9 if arm=="J3" else 1.}))
    assert qualification(rr,cfg)["local_capacity_passed"]
    for r in rr:
        if r["arm"]=="J3" and r["case"]<3:r["accepted"]=False
    assert not qualification(rr,cfg)["local_capacity_passed"]
    with pytest.raises(ValueError):qualification(rr[:-1],cfg)

def test_bad_right_seam_cannot_hide_in_means():
    cfg=config();m=dict(joins=[dict(root_step_m=1,root_turn_rad=1,foot_step_max_m=1,peak_jerk_mps3=1) for _ in range(2)],
        boundary_errors=[dict(root_position_m=1,root_velocity_mps=1,root_acceleration_mps2=1) for _ in range(2)],
        root_orientation_rad=1,pose_error_rad=1,upperbody_pose_rad=1,activity_mps=1,root_rmse_m=1,feet_rmse_m=1)
    bad=json.loads(json.dumps(m));bad["joins"][0]["root_step_m"]=.5;bad["joins"][1]["root_step_m"]=1.1
    assert not protected(bad,m,cfg)["both_seams_nonregression"]
    assert not protected(m,m,cfg)["root_improved"]


def test_manifest_absolute_relative_and_outside_paths(tmp_path):
    from dunhuang.research.root_pose_r15.common import manifest_ref
    from dunhuang.io import checked,sha
    file=tmp_path/"trace.npz";file.write_bytes(b"frozen trace")
    hashes={"trace.npz":sha(file)}
    a=manifest_ref(file,hashes,tmp_path);b=manifest_ref("trace.npz",hashes,tmp_path)
    assert a==b and checked(a)==file
    with pytest.raises(ValueError):manifest_ref(tmp_path.parent/"outside.npz",hashes,tmp_path)
    with pytest.raises(KeyError):manifest_ref("unknown.npz",hashes,tmp_path)
    file.write_bytes(b"changed trace")
    with pytest.raises(ValueError):checked(a)
