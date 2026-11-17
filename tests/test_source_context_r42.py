import numpy as np
import pytest
from dunhuang.research.source_context_r42.contracts import observed_anchor,window_starts,role_check


def test_hidden_body_and_root_cannot_change_observed_anchor():
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(42);r=Rotation.from_rotvec(rng.normal(scale=.01,size=(256*22,3))).as_matrix().reshape(256,22,3,3);raw=np.zeros((256,139));raw[:,7:]=r[...,:2,:].reshape(256,132);raw[:,4:7]=np.arange(256)[:,None]*.001;other=raw.copy();other[104:152]=rng.normal(size=(48,139));base=raw.copy()
    assert np.array_equal(observed_anchor(raw,base,[104,152],True),observed_anchor(other,base,[104,152],True))
    base[:,4:7]=rng.normal(size=(256,3));assert np.array_equal(observed_anchor(raw,base,[104,152],False)[:,4:7],base[:,4:7])


def test_windows_are_continuous_real_ranges_and_recordings_never_cross_roles():
    assert window_starts(256,True)==[0]
    assert len(window_starts(1024,True))==12 and window_starts(1024,False)==[0,384,768]
    with pytest.raises(ValueError):window_starts(255,True)
    rows=[dict(recording_uid='a',role='train'),dict(recording_uid='b',role='held')];assert role_check(rows)['recordings']==2
    with pytest.raises(ValueError):role_check(rows+[dict(recording_uid='a',role='held')])


def test_context_excludes_unknown_contact_root_and_hidden_frame_observations():
    import torch
    from dunhuang.research.coverage_r34.context_contract import context
    x=torch.zeros(2,256,139);x[:,104:152]=float('nan');cp=context(x,[104,152]);cp.valid[:,:,:7]=False;cp.values[:,:,:7]=float('nan');cp.validate();encoded=cp.encoded_input();assert torch.isfinite(encoded).all();assert not ((cp.frame_index>=104)&(cp.frame_index<152)).any()
    cp.frame_index[0,0]=110
    with pytest.raises(ValueError):cp.validate()
