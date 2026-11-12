import numpy as np
import pytest
import torch
from dunhuang.generation.root_contact import root_contact_features,RootContactAdapter,CHANNELS
from dunhuang.generation.event_music_adapter import EventMusicAdapter

def reference():
    x=torch.zeros(2,150,151);x[:,:,7:]=torch.tensor([1,0,0,0,1,0]).repeat(24)
    return x

def test_unknown_root_is_not_a_stationary_observation():
    x=reference();mask=torch.zeros_like(x,dtype=torch.bool)
    a=root_contact_features(x,mask)
    x[:,:,:]=float('nan')
    b=root_contact_features(x,mask)
    assert a.shape==(2,150,CHANNELS) and torch.equal(a,b) and not a.any()
    with pytest.raises(ValueError):root_contact_features(x,torch.ones_like(mask))

def test_root_translation_origin_invariance_and_derivative_units():
    x=reference();x[:,:,4]=torch.arange(150)/30
    m=torch.ones_like(x,dtype=torch.bool)
    a=root_contact_features(x,m)
    y=x.clone();y[:,:,4:7]+=torch.tensor([10.,-3.,4.])
    b=root_contact_features(y,m)
    torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-4)
    assert torch.allclose(a[:,1:,3],torch.ones(2,149),atol=1e-5)
    assert not a[:,:,9:17].any()

def test_zero_new_branch_preserves_existing_adapter_exactly():
    torch.manual_seed(4);base=EventMusicAdapter().eval()
    new=RootContactAdapter().eval();new.load_state_dict(base.state_dict(),strict=False)
    geometry=torch.randn(2,150,153);music=torch.randn(2,150,512);root=torch.randn(2,150,CHANNELS)
    with torch.no_grad():
        assert torch.equal(base(geometry,music),new(geometry,music,root=root))
        base.output.weight.fill_(.01);new.load_state_dict(base.state_dict(),strict=False)
        assert torch.equal(base(geometry,music),new(geometry,music,root=root))

def test_root_reference_changes_execution_identity_without_changing_legacy_plan_ids():
    from dunhuang.planning.event_plan import candidate_identity
    event=dict(event_uid='e',reference=dict(sha256='a'*64),source_interval=[0,150],
        music_interval=[0,150],time_warp=[[0,0],[150,150]])
    old=candidate_identity(event)
    event['root_observation']=dict(sha256='b'*64,path='private')
    assert candidate_identity(event)!=old

def test_source_root_masks_bind_hash_clock_and_overlap(tmp_path):
    import json
    from dunhuang.io import sha
    from dunhuang.generation.root_contact import build_reference
    n=225;x=np.zeros((n,151),np.float32)
    x[:,7:]=np.tile([1,0,0,0,1,0],24);x[:,4]=np.arange(n)/30
    path=tmp_path/'native.npy';np.save(path,x)
    mask=tmp_path/'observed.npy';np.save(mask,np.ones_like(x,bool))
    ref=lambda p:dict(path=str(p),sha256=sha(p))
    e=dict(schema='source_root_observation_v1',source_sha256=sha(path),fps=30,units='m',up_axis='y',
        provenance='test observed fixture',observed=ref(mask),contact_semantics='UNVERIFIED_GEOMETRIC_PROXY')
    evidence=tmp_path/'root.json';evidence.write_text(json.dumps(e))
    event=dict(source_interval=[0,n],music_interval=[0,n],time_warp=[[0,0],[n,n]],reference=ref(path),root_observation=ref(evidence))
    values,audit=build_reference(dict(frames=n,events=[event]),n)
    assert np.array_equal(values[0,75:],values[1,:75])
    assert audit['observed_root_frames']==n and audit['independent_static_contact_confidence']==0
    e['source_sha256']='b'*64;evidence.write_text(json.dumps(e));event['root_observation']=ref(evidence)
    with pytest.raises(ValueError,match='provenance'):build_reference(dict(frames=n,events=[event]),n)


def test_root_body_squared_energy_requires_signed_cross_term():
    from dunhuang.research.root_diagnosis import decompose
    x=reference()[0].numpy();t=np.arange(150)/30
    x[:,4]=t**3
    fk=np.zeros((150,24,3),np.float32)
    result=decompose(x,fk)['squared_component_mean']
    assert result['world']==0 and result['root_translation']>0 and result['root_relative']>0
    assert result['signed_cross_term']<0
    assert result['reconstruction_error']<1e-8
