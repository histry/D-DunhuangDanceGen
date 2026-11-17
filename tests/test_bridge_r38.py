import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.research.bridge_r38.geometry import observed_bridge,polynomial

def test_hidden_gap_poison_cannot_change_geometry_or_observed_body():
    t=np.arange(256)*.002;R=np.repeat(Rotation.from_rotvec(np.c_[t,t*0,t*0]).as_matrix()[:,None],22,axis=1);gap=[104,152]
    a=R.copy();a[104:152]=np.nan;b=R.copy();b[104:152]=Rotation.from_rotvec([2,0,0]).as_matrix()
    x=observed_bridge(a,gap);y=observed_bridge(b,gap)
    assert np.array_equal(x,y)
    assert np.array_equal(x[:104],R[:104]) and np.array_equal(x[152:],R[152:])
    assert np.max(abs(np.linalg.det(x)-1))<1e-10
    assert np.max(abs(x-R))<1e-10

def test_root_bridge_recovers_known_quadratic_without_gap_truth():
    t=np.arange(256)/30;y=np.c_[t,t*t,t*.3];x=y.copy();x[104:152]=np.nan
    assert np.allclose(polynomial(x,[104,152]),y,atol=1e-10)
    x[103]=np.nan
    with pytest.raises(ValueError):polynomial(x,[104,152])

def test_short_support_is_rejected():
    R=np.broadcast_to(np.eye(3),(20,22,3,3))
    with pytest.raises(ValueError):observed_bridge(R,[2,15])

def test_source_masks_keep_known_pose_but_not_cross_cut_derivative():
    from dunhuang.research.bridge_r38.source import feature_arrays,packet,SegmentedReference
    R=np.broadcast_to(np.eye(3),(256,22,3,3)).copy();R[104:152]=np.nan;ids=np.zeros(256,int);ids[200:]=1
    f,p,v,s=feature_arrays(R,[104,152],ids)
    assert p[152].all() and not v[152].any() and p[200].all() and not v[200].any()
    assert not p[104:152].any() and np.isfinite(f).all()
    pkt=packet(R[None],[104,152],['test'],ids[None],device='cpu');model=SegmentedReference();model.set_reference(**pkt.values)
    before=model.context[0].detach().clone();R[104:152]=Rotation.from_rotvec([1.,0,0]).as_matrix()
    model.set_reference(**packet(R[None],[104,152],['test'],ids[None],device='cpu').values)
    assert __import__('torch').equal(before,model.context[0])

def test_ordered_music_embedding_responds_to_reversal_and_masks_padding():
    import torch
    from dunhuang.research.bridge_r38.content import OrderedRetrieval
    torch.manual_seed(8);model=OrderedRetrieval().eval();music=torch.randn(2,32,50);x=torch.randn(2,32,274);mask=torch.ones_like(x,dtype=torch.bool);valid=torch.ones(2,32,dtype=torch.bool)
    with torch.no_grad():
        a=model.encode(music,x,mask,valid)[0];b=model.encode(music.flip(1),x,mask,valid)[0]
        changed=x.clone();changed[:,:,-2:]=999;c=model.encode(music,changed,mask,valid)[0:2]
        d=model.encode(music,x,mask,valid)[0:2]
    assert not torch.allclose(a,b,atol=1e-6)
    assert torch.equal(c[0],d[0]) and torch.equal(c[1],d[1])

def test_actual_multirecording_positive_schedule_has_no_test_rows():
    from dunhuang.research.bridge_r38.content import schedule,relations
    rows=[]
    for role in ('train','val'):
        for audio in ('a','b'):
            for t in (0,300):
                for rec in ('one','two'):
                    rows.append(dict(role=role,audio_sha256=audio,frames=120,audio_frame_interval=[t,t+120],raw_motion_sha256=audio+rec,paired_clock_verified=True))
    result=schedule(rows,3,4)
    for ids in result:
        assert all(rows[i]['role']=='train' for i in ids)
        p,n=relations([rows[i] for i in ids]);assert (p.sum(1)==2).all() and n.any(1).all()
