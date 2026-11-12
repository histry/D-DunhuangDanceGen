"""Fail-closed generation targets. Retrieval descriptors are never train targets."""
import numpy as np
import soundfile as sf
from dunhuang.io import checked
from .sources import read

def inspect(assets,require_features=True):
    from dunhuang.generation.targets import phase_target
    validated={};feature_cache={}
    def verify(reference):
        key=(reference['path'],reference['sha256'])
        if key not in validated:validated[key]=checked(reference)
        return validated[key]
    manifest=read(verify(assets['paired_generation_manifest']))
    if manifest.get('schema')!='paired_generation_manifest_v2':
        raise ValueError('V2 observed/derived masks and extraction receipts required')
    if manifest['checkpoint_sha256']!=assets['checkpoint']['sha256'] or manifest['scaler_sha256']!=assets['scaler']['sha256']:
        raise ValueError('Paired manifest bound to another checkpoint/scaler')
    with np.load(verify(assets['scaler'])) as s:scale,bias=s['scale'],s['bias']
    roles={};identities={};windows=[];blockers=set();seen=set()
    split_proof=None
    if manifest.get('recording_independence_verified') is not True:
        blockers.add('RECORDING_INDEPENDENCE_UNVERIFIED')
    elif not manifest.get('split_attestation'):
        blockers.add('MISSING_SPLIT_ATTESTATION')
    else:
        split_proof=read(verify(manifest['split_attestation']))
        if split_proof.get('status')!='verified' or not split_proof.get('method'):
            raise ValueError('Split attestation lacks verification method')
    if manifest.get('blind_test_claimed'):raise ValueError('Previously exposed public data is not a fresh blind test')
    for row in manifest['rows']:
        uid=row['window_uid']
        if uid in seen:raise ValueError('Duplicate paired window')
        seen.add(uid);group=row['source_group'];role=row['role']
        if role not in ('train','val','test'):raise ValueError('Explicit role required')
        if group in roles and roles[group]!=role:raise ValueError('Paired source leakage')
        roles[group]=role
        if split_proof is not None:
            expected=dict(source_group=group,role=role,recording_uid=row.get('recording_uid'),
                choreography_uid=row.get('choreography_uid'),music_uid=row['music_uid'])
            if (not expected['recording_uid'] or not expected['choreography_uid'] or
                split_proof['sources'].get(row['raw_motion']['sha256'])!=expected):
                raise ValueError('Verified split does not bind source/group/role identities')
        for kind,identity in [('audio',row['audio']['sha256']),('motion',row['raw_motion']['sha256']),
                              ('music',row['music_uid']),('choreography',row.get('choreography_uid')),
                              ('recording',row.get('recording_uid'))]:
            if identity:
                key=(kind,identity)
                if key in identities and identities[key]!=role:raise ValueError('Associated identity crosses roles')
                identities[key]=role
        raw=verify(row['raw_motion']);audio=verify(row['audio'])
        native=np.load(verify(row['motion']),allow_pickle=False)
        if native.dtype!=np.float32 or native.shape!=(150,151) or not np.isfinite(native).all():
            raise ValueError('Real finite FP32 150x151 target required')
        masks={k:np.load(verify(row[k]),allow_pickle=False) for k in ('observed','derived','supervision')}
        if any(v.dtype!=np.bool_ or v.shape!=(150,151) for v in masks.values()):raise ValueError('Boolean channel masks required')
        observed,derived,supervised=[masks[k] for k in ('observed','derived','supervision')]
        if np.any(observed&derived) or np.any(supervised&~(observed|derived)):raise ValueError('Unknown channels cannot be supervised')
        expected=24 if row['dataset']=='aistpp' else 22 if row['dataset']=='finedance' else None
        if expected is None or row['observed_body_joints']!=expected:raise ValueError('Unknown body observation contract')
        if observed[:,:4].any() or not observed[:,4:7+6*expected].all() or observed[:,7+6*expected:].any():
            raise ValueError('Source observations fabricated or missing')
        if not derived[:,:4].all() or derived[:,4:].any() or row['contact_policy']!='official_edge_forward_displacement_0.01m_last1':
            raise ValueError('Derived contact provenance differs')
        if not supervised.all():blockers.add('INCOMPLETE_151D_SUPERVISION')
        target,_=phase_target(native,scale,bias)
        if target is None:blockers.add('TARGET_OUTSIDE_FROZEN_SUPPORT')
        start=row['start_frame']/30+row['audio_offset_seconds'];info=sf.info(audio)
        if row['fps']!=30 or row['frames']!=150 or row['start_frame']<0 or not np.isfinite(start):
            raise ValueError('Physical paired clock differs')
        if start<0 or start+5>info.frames/info.samplerate+1e-9:raise ValueError('Audio crop requires padding/truncation')
        features=None
        if require_features and (not row.get('jukebox') or not row.get('jukebox_receipt')):
            blockers.add('MISSING_REAL_JUKEBOX')
        elif require_features:
            receipt=read(verify(row['jukebox_receipt']))
            if (receipt['audio_sha256']!=row['audio']['sha256'] or receipt['layer']!=66 or receipt['fps']!=30
                or receipt['edge_commit']!=assets['edge_commit'] or receipt['window_uid']!=uid
                or receipt['sample_start']!=round(start*info.samplerate)
                or receipt['sample_end']!=round((start+5)*info.samplerate)
                or receipt['features_sha256']!=row['jukebox']['sha256']):
                raise ValueError('Jukebox provenance/clock mismatch')
            key=(row['jukebox']['path'],row['jukebox']['sha256'])
            if key not in feature_cache:feature_cache[key]=np.load(verify(row['jukebox']),allow_pickle=False)
            features=feature_cache[key]
            if features.dtype!=np.float32 or features.shape!=(150,4800) or not np.isfinite(features).all() or not features.any():
                raise ValueError('Real 150x4800 Jukebox required')
        if not row.get('alignment_attestation'):blockers.add('UNVERIFIED_AUDIO_OFFSET')
        else:
            evidence=read(verify(row['alignment_attestation']))
            if (evidence.get('status')!='verified' or evidence.get('window_uid')!=uid or
                evidence.get('audio_sha256')!=row['audio']['sha256'] or
                evidence.get('motion_sha256')!=row['raw_motion']['sha256'] or
                evidence.get('audio_offset_seconds')!=row['audio_offset_seconds'] or not evidence.get('method')):
                raise ValueError('Alignment attestation does not bind this source/time')
        if role!='test' and target is not None and (not require_features or features is not None):windows.append((group,row['start_frame'],target,features,native))
    if not any(roles[r[0]]=='train' for r in windows) or not any(roles[r[0]]=='val' for r in windows):
        blockers.add('MISSING_DISJOINT_TRAIN_VAL')
    report=dict(schema='paired_generation_preflight_v2',rows=len(manifest['rows']),known_identity_isolation=True,
        blockers=sorted(blockers),ready=not blockers,training_steps=0,
        full_motion_qualified=False,music_qualified=False,roles=roles)
    if not require_features:report['stage']='TARGETS_CLOCK_IDENTITIES_ONLY_NO_JUKEBOX'
    return ([] if blockers else windows),roles,report
