"""Verified AIST clocks, frozen SDMR observations, no Test feature extraction."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write,pinned
from dunhuang.planning.time_map import motion_features74

def read(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

def metadata(assets):
    manifest=read(checked(assets['paired_generation_manifest']))
    checked(manifest['split_attestation'])
    identities={}
    for r in manifest['rows']:
        for key in ('source_group','recording_uid','choreography_uid','music_uid'):
            identity=(key,r[key])
            if identity in identities and identities[identity]!=r['role']:raise ValueError('Associated source crosses roles')
            identities[identity]=r['role']
        if r['pairing_evidence']!='official_record_and_measured_refined_video_waveform' or r['fps']!=30 or r['frames']!=150:
            raise ValueError('Registered pairing/clock differs')
    if not manifest['recording_independence_verified'] or manifest['blind_test_claimed']:
        raise ValueError('Verified identities and exposed-data status required')
    return manifest


def window_motion_features(native,start,frames=150):
    # Derivatives precede window selection; do not create a new numerical origin per halo.
    if type(start) is not int or start<0 or start+frames>len(native):
        raise ValueError('Continuous source support required')
    values,mask=motion_features74(native)
    return values[start:start+frames].copy(),mask[start:start+frames].copy()

def prepare(assets,global_ref,sync_ref,out):
    out=Path(out);out.mkdir();(out/'audio').mkdir();(out/'windows').mkdir()
    manifest=metadata(assets)
    runtime=None
    if assets.get('audio_runtime'):
        runtime=read(checked(assets['audio_runtime']))
        if runtime.get('schema')!='isolated_audio_runtime_v1' or runtime.get('version')!='0.5.0.post1':
            raise ValueError('Unregistered isolated audio runtime')
        checked(runtime['wheel'])
        for item in runtime['files']:checked(item)
        sys.path.insert(0,str(Path(runtime['import_root']).resolve()))
    import soxr
    if runtime and soxr.__version__!=runtime['version']:raise ValueError('Imported resampler differs')
    rows=sorted([r for r in manifest['rows'] if r['role'] in ('train','val')],key=lambda r:r['window_uid'])
    legacy=pinned(assets['legacy_root'],assets['legacy_commit']);sys.path.insert(0,str(legacy))
    from evaluation.paired_alignment_observations_v3 import full_audio_observations,CONTRACT
    global_payload=torch.load(checked(global_ref),map_location='cpu',weights_only=True)
    sync_payload=torch.load(checked(sync_ref),map_location='cpu',weights_only=True)
    if sync_payload.get('global_checkpoint_sha256')!=global_ref['sha256'] or sync_payload['norm']!=global_payload['norm']:
        raise ValueError('Existing global/sync bindings differ')
    from .model import RetrievalModel
    global_model=RetrievalModel().eval();global_model.load_state_dict(global_payload['state'])
    sync_model=RetrievalModel().eval();sync_model.load_state_dict(sync_payload['state'])
    for branch in ('music','motion'):
        if any(not torch.equal(v,getattr(sync_model,branch).state_dict()[k]) for k,v in getattr(global_model,branch).state_dict().items()):
            raise ValueError('Global/sync encoder identities differ')
    norm=global_payload['norm'];audio_cache={};motion_cache={};features={k:[] for k in ('music','motion','mm','dm','energy_music','energy_motion')}
    for i,r in enumerate(rows):
        e=read(checked(r['alignment_attestation']))
        if (e['status']!='verified' or e['window_uid']!=r['window_uid'] or e['audio_sha256']!=r['audio']['sha256']
            or e['motion_sha256']!=r['raw_motion']['sha256'] or e['audio_offset_seconds']!=r['audio_offset_seconds']):
            raise ValueError('Source origin evidence differs')
        key=r['audio']['sha256']
        if key not in audio_cache:
            values,mask,receipt=full_audio_observations(checked(r['audio']))
            np.savez_compressed(out/'audio'/(key+'.npz'),values=values,mask=mask)
            write(out/'audio'/(key+'.json'),receipt);audio_cache[key]=(values,mask)
        values,mask=audio_cache[key]
        positions=r['start_frame']+30*r['audio_offset_seconds']+np.arange(15,135)
        left=np.floor(positions).astype(int);right=np.ceil(positions).astype(int)
        if min(left)<0 or max(right)>=len(values):raise ValueError('Actual audio halo unavailable; no padding')
        alpha=(positions-left)[:,None]
        music=values[left]*(1-alpha)+values[right]*alpha;mm=mask[left]&mask[right]
        native=np.load(checked(r['motion']));observed=np.load(checked(r['observed']))
        if not observed[:,7:139].all():raise ValueError('Unobserved body rotations cannot become alignment truth')
        raw_key=r['raw_motion']['sha256']
        if raw_key not in motion_cache:
            from dunhuang.data.conversion import decode
            full_native,_=decode(checked(r['raw_motion']),r['dataset'])
            continuous,continuous_mask=motion_features74(full_native)
            motion_cache[raw_key]=(full_native,continuous,continuous_mask)
        full_native,continuous,continuous_mask=motion_cache[raw_key]
        start=r['start_frame'];stop=start+150
        if stop>len(full_native) or not np.array_equal(full_native[start:stop,7:],native[:,7:]):
            raise ValueError('Cached native window differs from its continuous raw recording')
        motion=continuous[start:stop].copy();dm=continuous_mask[start:stop].copy()
        arrays=dict(music=music.astype(np.float32),motion=motion,mm=mm,dm=dm,
            energy_music=music[:,0].astype(np.float32),energy_motion=motion[:,66])
        file=out/'windows'/('%05d.npz'%i);np.savez_compressed(file,**arrays)
        for k in features:features[k].append(arrays[k])
        if (i+1)%200==0:print('CLOCK_DATA',i+1,len(rows),flush=True)
    arrays={k:np.stack(v) for k,v in features.items()}
    for name,mask in [('music','mm'),('motion','dm')]:
        arrays[name]=(((arrays[name]-np.asarray(norm[name]['mean']))/np.asarray(norm[name]['std']))*arrays[mask]).astype(np.float32)
    np.savez_compressed(out/'observations.npz',**arrays)
    write(out/'manifest.json',dict(schema='verified_clock_alignment_population_v1',rows=rows,
        paired_manifest=assets['paired_generation_manifest'],feature_contract=CONTRACT,
        audio_runtime=assets.get('audio_runtime'),resampler_version=soxr.__version__,libsoxr_version=soxr.__libsoxr_version__,
        extractor_commit=assets['legacy_commit'],extractor_sha256=sha(legacy/'evaluation/paired_alignment_observations_v3.py'),
        global_checkpoint=global_ref,sync_checkpoint=sync_ref,norm=norm,
        train_rows=sum(r['role']=='train' for r in rows),val_rows=sum(r['role']=='val' for r in rows),
        test_rows_extracted=0,negative_shift_labels_in_inputs=False,clock='source30Hz then real120-frame core and150-frame halo',
        motion_derivative_context='entire decoded recording before window/halo selection; initial source increment copied only at true recording origin'))
    return rows,arrays

def encode(model,arrays,out,device):
    banks={k:[] for k in ('mt','mv','dt','dv','mg','dg','tcn')}
    with torch.no_grad():
        for a in range(0,len(arrays['music']),32):
            m=torch.as_tensor(arrays['music'][a:a+32],device=device)
            d=torch.as_tensor(arrays['motion'][a:a+32],device=device)
            mm=torch.as_tensor(arrays['mm'][a:a+32],device=device)
            dm=torch.as_tensor(arrays['dm'][a:a+32],device=device)
            mg,mt,mv=model.music(m,mm);tcn,_=model.motion.temporal_features(d[:,15:135],dm[:,15:135])
            ds=torch.stack([d[:,15+s:135+s] for s in (-15,-8,0,8,15)],1)
            masks=torch.stack([dm[:,15+s:135+s] for s in (-15,-8,0,8,15)],1)
            dg,dt,dv=model.motion(ds.flatten(0,1),masks.flatten(0,1))
            for k,v in dict(mt=mt,mv=mv,dt=dt.reshape(len(m),5,60,128),dv=dv.reshape(len(m),5,60),
                mg=mg,dg=dg.reshape(len(m),5,256)[:,2],tcn=tcn).items():banks[k].append(v.cpu().numpy())
    result={k:np.concatenate(v) for k,v in banks.items()}
    np.savez_compressed(out,**result)
    return result
