import pickle,copy,json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.data.conversion import decode,normalized_window
from dunhuang.data.sources import validate_links,relative
from dunhuang.data.paired_manifest import inspect
from dunhuang.contracts import matrices,sixd

def test_aist_inverse_scale_exact_decimation_and_root_height(tmp_path):
    r=np.zeros((302,72),np.float32);r[:,0]=np.linspace(0,1,302)
    root=np.stack([np.arange(302),np.full(302,100),np.zeros(302)],1).astype(np.float32)
    path=tmp_path/'a.pkl';path.write_bytes(pickle.dumps(dict(smpl_poses=r,smpl_trans=root,smpl_scaling=np.array([100]))))
    native,observed=decode(path,'aistpp')
    assert native.shape==(151,151) and observed[4:].all() and not observed[:4].any()
    assert np.allclose(native[:,4],np.arange(0,302,2)/100) and np.allclose(native[:,5],1)
    _,n,v=normalized_window(native,1,np.ones(151),np.zeros(151))
    assert n[0,4]==0 and n[0,6]==1 # Native Y height -> EDGE Z, preserved.
    assert np.all(n[-1,:4]==1) and native[-1,0]==0

def test_finedance_22_not_fake_24_and_row_rot6d(tmp_path):
    rotations=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix()
    row=sixd(np.broadcast_to(rotations,(150,52,3,3)),row=True)
    x=np.concatenate([np.ones((150,3)),row.reshape(150,312)],1).astype(np.float32)
    path=tmp_path/'fine.npy';np.save(path,x)
    native,observed=decode(path,'finedance')
    assert np.allclose(matrices(native[:,7:].reshape(-1,24,6))[:,:22],rotations,atol=1e-6)
    assert not observed[-12:].any() and observed[4:139].all()
    x[:,3:9]=0;np.save(path,x)
    with pytest.raises(ValueError,match='Degenerate'):decode(path,'finedance')

def rows():
    return [dict(pair_uid=str(i),group_uid=str(i),purpose_role=role,dataset_id='aistpp',
        motion_asset=dict(sha256='motion'+str(i)),audio_asset=dict(sha256='audio'+str(i)),
        music_id='song'+str(i)) for i,role in enumerate(('train','val','test'))]

@pytest.mark.parametrize('identity',['music_id','conservative_family_key','normalized_song_name'])
def test_connected_identity_cannot_cross_splits(identity):
    r=rows();r[0][identity]='same';r[1][identity]='same'
    with pytest.raises(ValueError,match='crosses roles'):validate_links(r)

def test_heldout_duplicate_audio_and_motion_rejected():
    for asset in ('audio_asset','motion_asset'):
        r=rows();r[2][asset]=r[0][asset]
        with pytest.raises(ValueError):validate_links(r)
    r=rows();r.append(dict(purpose_role='excluded_official_ignore',official_ignore=True))
    assert validate_links(r)['groups']==3

def test_registry_path_cannot_escape_data_root():
    with pytest.raises(ValueError):relative(dict(path='/other/no_dataset/file'))
    with pytest.raises(ValueError):relative(dict(path='/x/assets/datasets/../../secret'))

def test_legacy_manifest_cannot_bypass_observation_contract(tmp_path):
    from dunhuang.io import sha
    path=tmp_path/'old.json';path.write_text(json.dumps(dict(rows=[],schema='legacy')))
    with pytest.raises(ValueError,match='V2'):inspect(dict(paired_generation_manifest=dict(path=str(path),sha256=sha(path))))

def test_numerical_support_does_not_clip_unknown_or_out_of_range_channels():
    x=np.zeros((150,151),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],24)
    x[:,5]=3
    _,target,valid=normalized_window(x,0,np.ones(151),np.zeros(151))
    assert target[0,6]==3 and not valid[:,6].any()
