import numpy as np
import pytest
from dunhuang.evaluation_benchmarks.edge_pfc import pfc
from dunhuang.evaluation_benchmarks.physdiff_metrics import metrics
from dunhuang.evaluation_benchmarks.contact_states import diagnose
from dunhuang.contracts import Q

def positions():
    x=np.zeros((20,24,3),np.float32)
    t=np.arange(20,dtype=np.float32)
    x[:,0,0]=t*t*.001
    x[:,[7,10,8,11],0]=t[:,None]*.01
    return x

def test_pfc_axes_displacement_and_undefined_acceleration():
    x=positions()
    a=pfc(x)['value']
    assert a>0 and a==pfc(x@Q.T,up_axis='z')['value']
    assert pfc(np.zeros_like(x))['value'] is None
    x[:,[7,10],0]=0
    assert pfc(x)['value']==0
    with pytest.raises(ValueError,match='30 FPS'):pfc(x,fps=60)

def test_mesh_tolerance_contact_population_and_no_fake_zero():
    joints=np.zeros((4,24,3))
    minimum=np.array([-.004,-.010,.004,.020])
    r=metrics(minimum,joints,0,contact_mask=np.ones((4,4),bool))
    assert r['penetrate_mm']==pytest.approx(2.5)
    assert r['float_mm']==pytest.approx(5.)
    assert r['skate_mm']==0 and r['phys_err_mm']==pytest.approx(7.5)
    empty=metrics(minimum,joints,0,contact_mask=np.zeros((4,4),bool))
    assert empty['skate_mm'] is None and empty['phys_err_mm'] is None
    assert metrics(None,joints,0)['status']=='MISSING_MESH'

def test_high_speed_cannot_self_declare_intentional_sliding():
    x=positions()
    out=diagnose(x,0,np.ones((20,4)))
    assert out['counts']['2']==0 and out['counts']['-1']==80
    assert out['unknown_horizontal_speed_p95_mps']>0
    with pytest.raises(ValueError,match='Verified'):
        diagnose(x,0,labels=np.full((20,4),2))

def test_lbs_rest_and_global_root_transform():
    import torch
    from dunhuang.evaluation_benchmarks.smpl_mesh import SMPLMesh
    from dunhuang.contracts import sixd
    from scipy.spatial.transform import Rotation
    mesh=SMPLMesh.__new__(SMPLMesh);mesh.device='cpu'
    mesh.template=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    mesh.rest=torch.zeros(24,3);mesh.weights=torch.zeros(3,24);mesh.weights[:,0]=1
    mesh.posedirs=torch.zeros(9,207);mesh.parents=[-1]+[0]*23
    x=np.zeros((2,151),np.float32);x[:,4:7]=[2,3,4];x[:,7:]=np.tile([1,0,0,0,1,0],24)
    r=Rotation.from_euler('y',.5).as_matrix();x[:,7:13]=sixd(r)
    v,j=mesh.forward(x)
    np.testing.assert_allclose(v.numpy(),np.broadcast_to(mesh.template.numpy()@r.T+[2,3,4],v.shape),atol=1e-6)
    np.testing.assert_allclose(j[:,0].numpy(),x[:,4:7],atol=1e-6)

def test_declared_contacts_use_native_ankle_ankle_toe_toe_order():
    x=np.zeros((4,24,3));x[:,:,1]=10
    x[:,8,1]=0
    declared=np.zeros((4,4));declared[:,1]=1
    result=diagnose(x,0,declared)
    assert result['foot_joint_indices']==[7,8,10,11]
    assert result['counts']['-1']==4
