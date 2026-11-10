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


def test_trajectory_guidance_controls_fk_derivatives(tmp_path):
    """Single moving joint with a rigid distal point; audited real FK is checked by b2.py."""
    pytest.importorskip("pytorch3d")
    from scipy.spatial.transform import Rotation
    from dunhuang.temporal import prepare
    from dunhuang.contracts import MotionAdapter, matrices, sixd
    from dunhuang.io import sha
    identity = np.eye(3)
    source = np.zeros((25,151),np.float32)
    source[:,7:] = np.tile(sixd(identity),24)
    channel = slice(7+6*16,13+6*16)
    angles = np.linspace(.35,.45,25)
    source[:,channel] = sixd(Rotation.from_rotvec(np.c_[angles*0,angles*0,angles]).as_matrix())
    file = tmp_path/'source.npy'
    np.save(file,source)
    class Normalizer:
        class scaler:
            scale_=torch.ones(151)
            min_=torch.zeros(151)
        def normalize(self,x):return x
        def unnormalize(self,x):return x
    anchor=dict(frame=70,source_frame=12,joints=[16],pose=source[12].tolist(),
                source=dict(path=str(file),sha256=sha(file)))
    guide=prepare(dict(anchors=[anchor],frames=150,trajectory_radius=12,trajectory_strength=.55),
                  Normalizer(),tmp_path,'cpu')
    rest=source[:1].copy()
    rest[:,channel]=sixd(identity)
    predicted=torch.from_numpy(np.tile(MotionAdapter.to_edge(rest),(1,150,1)))
    soft=guide(predicted,1.).numpy()[0]
    hard=predicted.numpy()[0].copy()
    hard[70,channel]=MotionAdapter.to_edge(source)[12,channel]
    def endpoint(motion):
        return matrices(motion[:,channel],row=True)@np.array([.35,0.,0.])
    assert np.array_equal(soft[58],predicted.numpy()[0,58])
    assert np.linalg.norm(endpoint(soft)[70]-endpoint(hard)[70])<np.linalg.norm(endpoint(predicted.numpy()[0])[70]-endpoint(hard)[70])
    for order in (1,2,3):
        assert abs(np.diff(endpoint(soft),n=order,axis=0)).max()<abs(np.diff(endpoint(hard),n=order,axis=0)).max()
