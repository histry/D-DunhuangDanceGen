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


def paired_fixture(tmp_path):
    from dunhuang.io import sha
    import soundfile as sf
    def ref(path):return dict(path=str(path),sha256=sha(path))
    scaler=tmp_path/'scaler.npz';np.savez(scaler,scale=np.ones(151,np.float32),bias=np.zeros(151,np.float32))
    rows=[]
    for index,role in enumerate(('train','val')):
        folder=tmp_path/str(index);folder.mkdir()
        raw=folder/'raw';raw.write_bytes(('source'+str(index)).encode())
        audio=folder/'audio.wav';sf.write(audio,np.ones(80000,np.float32)*(.1+index*.01),16000,subtype='FLOAT')
        motion=np.zeros((150,151),np.float32);motion[:,:4]=1;motion[:,5]=.5
        motion[:,7:]=np.tile([1,0,0,0,1,0],24);np.save(folder/'motion.npy',motion)
        observed=np.ones((150,151),bool);observed[:,:4]=False
        derived=~observed
        for name,v in [('observed',observed),('derived',derived),('supervision',observed|derived)]:
            np.save(folder/(name+'.npy'),v)
        np.save(folder/'jukebox.npy',np.ones((150,4800),np.float32)*.1)
        row=dict(window_uid=str(index),source_group='group'+str(index),role=role,dataset='aistpp',
            recording_uid='recording'+str(index),choreography_uid='choreography'+str(index),
            audio=ref(audio),raw_motion=ref(raw),music_uid='music'+str(index),fps=30,frames=150,
            start_frame=0,audio_offset_seconds=0,observed_body_joints=24,
            contact_policy='official_edge_forward_displacement_0.01m_last1')
        for name in ('motion','observed','derived','supervision','jukebox'):row[name]=ref(folder/(name+'.npy'))
        receipt=dict(audio_sha256=sha(audio),layer=66,fps=30,edge_commit='edge',window_uid=str(index),
            sample_start=0,sample_end=80000,features_sha256=row['jukebox']['sha256'])
        path=folder/'receipt.json';path.write_text(json.dumps(receipt));row['jukebox_receipt']=ref(path)
        attestation=dict(status='verified',window_uid=str(index),audio_sha256=sha(audio),
            motion_sha256=sha(raw),audio_offset_seconds=0,method='synthetic unit fixture only')
        path=folder/'alignment.json';path.write_text(json.dumps(attestation));row['alignment_attestation']=ref(path)
        rows.append(row)
    split_proof=dict(status='verified',method='synthetic unit fixture only',sources={
        r['raw_motion']['sha256']:{k:r[k] for k in ('source_group','role','recording_uid','choreography_uid','music_uid')} for r in rows})
    split_path=tmp_path/'split.json';split_path.write_text(json.dumps(split_proof))
    manifest=dict(split_attestation=ref(split_path),schema='paired_generation_manifest_v2',checkpoint_sha256='checkpoint',
        scaler_sha256=sha(scaler),rows=rows,recording_independence_verified=True,blind_test_claimed=False)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    assets=dict(paired_generation_manifest=ref(path),checkpoint=dict(sha256='checkpoint'),
                scaler=ref(scaler),edge_commit='edge')
    return assets,manifest,path

def resave(assets,manifest,path):
    from dunhuang.io import sha
    path.write_text(json.dumps(manifest));assets['paired_generation_manifest']['sha256']=sha(path)

def test_real_shapes_provenance_and_preflight_without_change_catalog(tmp_path):
    from dunhuang.research.legacy.paired_training import train
    assets,manifest,path=paired_fixture(tmp_path)
    windows,roles,report=inspect(assets)
    assert report['ready'] and len(windows)==2 and set(roles.values())=={'train','val'}
    train(assets,tmp_path/'preflight',1,task='paired',preflight_only=True)
    status=json.loads((tmp_path/'preflight/status.json').read_text())
    assert status['status']=='PREFLIGHT_READY_NOT_QUALIFIED' and status['training_steps']==0

def test_release_declared_offset_never_promoted_to_verified_training(tmp_path):
    assets,manifest,path=paired_fixture(tmp_path)
    manifest['rows'][0]['alignment_attestation']=None
    manifest['recording_independence_verified']=False;resave(assets,manifest,path)
    windows,_,report=inspect(assets)
    assert not windows and not report['ready']
    assert {'UNVERIFIED_AUDIO_OFFSET','RECORDING_INDEPENDENCE_UNVERIFIED'}<=set(report['blockers'])

@pytest.mark.parametrize('change',['wrong_layer','wrong_clock','fake_observation','missing_hands'])
def test_paired_manifest_rejects_false_observation_and_feature_clock(tmp_path,change):
    from dunhuang.io import sha
    assets,manifest,path=paired_fixture(tmp_path);row=manifest['rows'][0]
    if change in ('wrong_layer','wrong_clock'):
        p=Path(row['jukebox_receipt']['path']);v=json.loads(p.read_text())
        v['layer' if change=='wrong_layer' else 'sample_start']=35 if change=='wrong_layer' else 1
        p.write_text(json.dumps(v));row['jukebox_receipt']['sha256']=sha(p)
    elif change=='fake_observation':
        row['dataset']='finedance';row['observed_body_joints']=22
    else:
        p=Path(row['supervision']['path']);v=np.load(p);v[:,-12:]=False;np.save(p,v)
        row['supervision']['sha256']=sha(p)
    resave(assets,manifest,path)
    if change=='missing_hands':
        windows,_,report=inspect(assets)
        assert not windows and 'INCOMPLETE_151D_SUPERVISION' in report['blockers']
    else:
        with pytest.raises(ValueError):inspect(assets)

def test_test_role_identity_leak_is_checked_before_skipping_fit(tmp_path):
    assets,manifest,path=paired_fixture(tmp_path)
    row=copy.deepcopy(manifest['rows'][0]);row.update(window_uid='test',role='test',source_group='new-test-group')
    manifest['rows'].append(row);resave(assets,manifest,path)
    with pytest.raises(ValueError):inspect(assets)


def test_boolean_claim_without_source_bound_split_receipt_not_train_ready(tmp_path):
    assets,manifest,path=paired_fixture(tmp_path)
    manifest.pop('split_attestation');resave(assets,manifest,path)
    windows,_,report=inspect(assets)
    assert not windows and 'MISSING_SPLIT_ATTESTATION' in report['blockers']


def test_pre_extraction_gate_cannot_bypass_training_feature_gate(tmp_path):
    assets,manifest,path=paired_fixture(tmp_path)
    for row in manifest['rows']:
        row['jukebox']=None;row['jukebox_receipt']=None
    resave(assets,manifest,path)
    windows,_,pre=inspect(assets,require_features=False)
    assert pre['ready'] and len(windows)==2 and pre['stage']=='TARGETS_CLOCK_IDENTITIES_ONLY_NO_JUKEBOX'
    windows,_,strict=inspect(assets)
    assert not windows and not strict['ready'] and 'MISSING_REAL_JUKEBOX' in strict['blockers']
