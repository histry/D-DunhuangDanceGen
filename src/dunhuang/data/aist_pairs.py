"""Complete source ledger, calibrated basic-dance index, and deduplicated features."""
import argparse,json,os,sys,time,pickle,shutil,subprocess,hashlib,concurrent.futures
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.io import checked,pinned,sha,write
from dunhuang.contracts import MotionAdapter
from .sources import read,ref,raw_path
from .conversion import decode,contacts,normalized_window
from .aist_authority import authority,identify,calibrate,POLICY

def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp');write(temporary,value);temporary.replace(path)

def fetch(url,path):
    import requests
    path=Path(path)
    if path.exists():return ref(path)
    temporary=path.with_suffix('.part')
    for attempt in range(2):
        try:
            with requests.Session() as session:
                session.trust_env=bool(attempt)
                with session.get(url,stream=True,timeout=(15,90)) as r:
                    r.raise_for_status();total=0;begun=time.monotonic()
                    with temporary.open('wb') as f:
                        for chunk in r.iter_content(1024*1024):
                            total+=len(chunk)
                            if total>64*1024**2 or time.monotonic()-begun>180:raise ValueError('Bounded video download exceeded')
                            f.write(chunk)
            with temporary.open('rb') as f:
                if b'ftyp' not in f.read(64):raise ValueError('Publisher response is not MP4')
            temporary.replace(path);return ref(path)
        except Exception:
            if temporary.exists():temporary.unlink()
            if attempt==1:raise


def run_calibration(assets,root,authroot,out,source_audio,workers=6,smoke=False):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(root);auth=authority(authroot);population=read(root/'population.json')
    rows=[r for r in population['records'] if r['dataset_id']=='aistpp']
    if smoke:
        picked={}
        for row in rows:
            if '_sBM_' in row['sequence_id']:picked.setdefault(row['purpose_role'],row)
        rows=[picked[role] for role in ('train','val','test')]
    (out/'calibration').mkdir();videos=root/'aistpp/refined_video';videos.mkdir(exist_ok=True)
    # Copy only registered music, verify bytes before signal comparison.
    for row in rows:
        target=raw_path(root,row['audio_asset'],required=False)
        if not target.exists():
            original=Path(source_audio)/(row['music_id']+'.wav')
            checked(dict(path=str(original),sha256=row['audio_asset']['sha256']))
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
        checked(dict(path=str(target),sha256=row['audio_asset']['sha256']))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    save(out/'protocol.json',dict(code_commit=commit,authority=ref(Path(authroot)/'authority.json'),
        population=ref(root/'population.json'),policy=POLICY,workers=workers,smoke=smoke,
        scope='all registered AIST source ledger; basic choreography only admitted',
        advanced_policy='excluded: official shared advanced choreography has no per-routine identity table',
        historical_roles_unchanged=True,blind_test_claimed=False,training_steps=0))
    def worker(row):
        name=row['sequence_id'];result=dict(pair_uid=row['pair_uid'],role=row['purpose_role'],source_group=row['group_uid'],
            raw_motion=row['motion_asset'],raw_audio=row['audio_asset'])
        try:
            identity=identify(row,auth);result.update(identity)
            if identity['status']!='REGISTERED_BASIC':return result
            raw=raw_path(root,row['motion_asset'])
            with raw.open('rb') as f:frames=len(pickle.load(f)['smpl_poses'])
            movie=videos/(name+'_'+hashlib.sha256(identity['video_url'].encode()).hexdigest()[:10]+'.mp4')
            video=fetch(identity['video_url'],movie)
            if identity.get('publisher_metadata') and movie.stat().st_size!=int(identity['publisher_metadata']['FILE_SIZE']):
                raise ValueError('Movie bytes differ from official full CSV size')
            value=calibrate(movie,raw_path(root,row['audio_asset']),frames)
            result.update(calibration=value,video=video,
                status='VERIFIED_PAIR' if value['verified'] else 'EXCLUDED_CLOCK_CALIBRATION')
            result['authority_sha256']=auth['identity']
            save(out/'calibration'/(name+'.json'),result)
        except Exception as error:
            result.update(status='EXCLUDED_TECHNICAL',error=repr(error))
            save(out/'calibration'/(name+'.json'),result)
        return result
    results=[];begun=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(worker,row):row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results)%10==0:
                counts={s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})}
                save(out/'progress.json',dict(completed=len(results),total=len(rows),counts=counts,seconds=time.monotonic()-begun))
                print('CALIBRATED',len(results),len(rows),counts,flush=True)
    order={r['pair_uid']:i for i,r in enumerate(rows)};results.sort(key=lambda r:order[r['pair_uid']])
    save(out/'source_ledger.json',results)
    counts={s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})}
    save(out/'status.json',dict(status='CALIBRATION_COMPLETE',counts=counts,total=len(rows),seconds=time.monotonic()-begun,
        optimizer_steps=0,full_motion_qualified=False,music_qualified=False))
    return out

def compile_index(assets,root,calibration,audit,out):
    root=Path(root);calibration=Path(calibration);out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    if read(calibration/'protocol.json').get('smoke'):raise ValueError('Smoke calibration cannot define a full training population')
    if read(calibration/'status.json')['counts'].get('EXCLUDED_TECHNICAL',0):
        raise ValueError('Resolve technical calibration failures before selecting population')
    ledger=read(calibration/'source_ledger.json');window_rows=read(Path(audit)/'windows.json')
    audit_report=read(Path(audit)/'report.json')
    if (audit_report['conversion_sha256']!=sha(Path(__file__).with_name('conversion.py')) or
        audit_report['scaler']['sha256']!=assets['scaler']['sha256']):raise ValueError('Representation audit changed')
    # Retain source-linked roles and enforce identities, without moving a source.
    known={};admitted={};duplicates={};excluded=[]
    for row in ledger:
        if row['status']!='VERIFIED_PAIR':continue
        rawsha=row['raw_motion']['sha256']
        keys=[('recording',row['recording_uid']),('choreography',row['choreography_uid']),
              ('audio',row['raw_audio']['sha256']),('motion',rawsha)]
        for key in keys:
            previous=known.get(key)
            if previous is not None and previous!=row['role']:raise ValueError('Verified identities cross historical roles: '+str(key))
            known[key]=row['role']
        if rawsha in duplicates:
            excluded.append(dict(pair_uid=row['pair_uid'],reason='duplicate_raw_motion',canonical=duplicates[rawsha]));continue
        duplicates[rawsha]=row['pair_uid'];admitted[row['pair_uid']]=row
    edge=pinned(assets['edge_root'],assets['edge_commit']);sys.path.insert(0,str(edge))
    import torch
    from vis import SMPLSkeleton
    torch.set_num_threads(4);skeleton=SMPLSkeleton('cpu')
    with np.load(checked(assets['scaler'])) as s:scale,bias=s['scale'],s['bias']
    maskdir=out/'masks';maskdir.mkdir()
    observed=np.ones((150,151),bool);observed[:,:4]=False
    derived=~observed
    for k,v in [('observed',observed),('derived',derived),('supervision',observed|derived)]:np.save(maskdir/(k+'.npy'),v)
    windows=[];proof={};rejected=[];source_windows={}
    for row in window_rows:
        if row['dataset']=='aistpp' and row['pair_uid'] in admitted and row['representation_eligible']:
            source_windows.setdefault(row['pair_uid'],[]).append(row)
    targets=out/'targets';targets.mkdir()
    for uid,rows in source_windows.items():
        source=admitted[uid];native,_=decode(raw_path(root,source['raw_motion']),'aistpp')
        native[:,:4]=contacts(MotionAdapter.to_edge(native),skeleton)
        sourcefile=ref(raw_path(root,source['raw_motion']));audiofile=ref(raw_path(root,source['raw_audio']))
        proof[sourcefile['sha256']]=dict(source_group=source['source_group'],role=source['role'],
            recording_uid=source['recording_uid'],choreography_uid=source['choreography_uid'],music_uid='audio:'+audiofile['sha256'])
        info=sf.info(audiofile['path']);offset=source['calibration']['audio_offset_seconds']
        for row in rows:
            start=row['start_frame'];audio_start=start/30+offset
            if audio_start<0 or audio_start+5>info.frames/info.samplerate:
                rejected.append(dict(window_uid=row['window_uid'],reason='insufficient_real_audio'));continue
            target,_,valid=normalized_window(native,start,scale,bias)
            if not valid.all():raise ValueError('Representable target changed')
            from dunhuang.generation.targets import phase_target
            if phase_target(target,scale,bias)[0] is None:
                rejected.append(dict(window_uid=row['window_uid'],reason='phase_roundtrip_outside_support'));continue
            stem=hashlib.sha256(row['window_uid'].encode()).hexdigest()[:24]
            np.save(targets/(stem+'.npy'),target)
            attestation=dict(status='verified',window_uid=row['window_uid'],audio_sha256=audiofile['sha256'],
                motion_sha256=sourcefile['sha256'],audio_offset_seconds=offset,
                method=source['calibration']['method'],source_calibration=ref(calibration/'calibration'/(source['sequence_id']+'.json')),
                maximum_residual_seconds=source['calibration']['maximum_residual_seconds'],
                authority_sha256=source['authority_sha256'],video=source['video'])
            save(targets/(stem+'.alignment.json'),attestation)
            result=dict(window_uid=row['window_uid'],dataset='aistpp',source_group=source['source_group'],role=source['role'],
                start_frame=start,fps=30,frames=150,audio_offset_seconds=offset,audio=audiofile,raw_motion=sourcefile,
                motion=ref(targets/(stem+'.npy')),observed_body_joints=24,contact_policy='official_edge_forward_displacement_0.01m_last1',
                music_uid='audio:'+audiofile['sha256'],recording_uid=source['recording_uid'],
                choreography_uid=source['choreography_uid'],alignment_attestation=ref(targets/(stem+'.alignment.json')),
                pairing_evidence='official_record_and_measured_refined_video_waveform',jukebox=None,jukebox_receipt=None)
            for k in ('observed','derived','supervision'):result[k]=ref(maskdir/(k+'.npy'))
            windows.append(result)
        print('INDEX_SOURCE',uid,len(rows),flush=True)
    save(out/'split_attestation.json',dict(status='verified',method='official camera-collapsed captures and workbook basic choreography; all shared music/capture/choreography identities checked across retained historical roles',
        sources=proof,scope='publisher genre-namespaced basic identities; not statistical independence or a blind test',
        source_ledger=ref(calibration/'source_ledger.json')))
    manifest=dict(schema='paired_generation_manifest_v2',checkpoint_sha256=assets['checkpoint']['sha256'],
        scaler_sha256=assets['scaler']['sha256'],recording_independence_verified=True,blind_test_claimed=False,
        split_attestation=ref(out/'split_attestation.json'),stage='verified_targets_before_jukebox',rows=windows)
    save(out/'manifest.json',manifest);prepared=dict(assets,paired_generation_manifest=ref(out/'manifest.json'))
    save(out/'assets.json',prepared);save(out/'excluded_windows.json',dict(rejected=rejected,duplicate_sources=excluded))
    from .paired_manifest import inspect
    _,_,report=inspect(prepared,require_features=False);save(out/'preflight.json',report)
    save(out/'status.json',dict(status='TARGET_INDEX_READY' if report['ready'] else 'TARGET_INDEX_BLOCKED',
        rows=len(windows),ready=report['ready'],training_steps=0))
    return out

def extract_features(assets,index,out):
    index=Path(index);out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    prepared=read(index/'assets.json')
    from .paired_manifest import inspect
    _,_,preflight=inspect(prepared,require_features=False)
    if not preflight['ready']:raise ValueError('Target/identity/time preflight blocked')
    manifest=read(checked(prepared['paired_generation_manifest']))
    edge=pinned(assets['edge_root'],assets['edge_commit']);sys.path.insert(0,str(edge))
    from data.audio_extraction.jukebox_features import extract
    import importlib.metadata
    cache={};result=[]
    for i,row in enumerate(manifest['rows']):
        info=sf.info(checked(row['audio']));start=round((row['start_frame']/30+row['audio_offset_seconds'])*info.samplerate)
        end=start+5*info.samplerate
        identity=dict(audio=row['audio']['sha256'],sample_start=start,sample_end=end,rate=info.samplerate,
            edge_commit=assets['edge_commit'],extractor=sha(edge/'data/audio_extraction/jukebox_features.py'),
            layer=66,fps=30,channel_policy='mono_mean_float32')
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        if key not in cache:
            folder=out/key;folder.mkdir()
            with sf.SoundFile(row['audio']['path']) as audio:
                if start<0 or end>len(audio):raise ValueError('No synthetic audio allowed')
                audio.seek(start);pcm=audio.read(end-start,dtype='float32',always_2d=True).mean(1)
            crop=folder/'audio.wav';sf.write(crop,pcm,info.samplerate,subtype='FLOAT')
            value,_=extract(str(crop),skip_completed=False,dest_dir=str(folder/'extract_cache'))
            value=np.asarray(value,np.float32)
            if value.shape!=(150,4800) or not np.isfinite(value).all() or not value.any():raise ValueError('Real Jukebox differs')
            np.save(folder/'jukebox.npy',value);cache[key]=dict(folder=folder,features=ref(folder/'jukebox.npy'),crop=ref(crop))
            save(folder/'producer.json',dict(identity,features_sha256=cache[key]['features']['sha256'],crop=cache[key]['crop'],
                jukemirlib_version=(importlib.metadata.version('jukemirlib')),training=False))
            print('EXTRACTED_UNIQUE',len(cache),'ROW',i,len(manifest['rows']),flush=True)
        hit=cache[key];receipt=dict(window_uid=row['window_uid'],audio_sha256=row['audio']['sha256'],
            sample_start=start,sample_end=end,sample_rate=info.samplerate,layer=66,fps=30,
            edge_commit=assets['edge_commit'],features_sha256=hit['features']['sha256'],
            extractor_sha256=identity['extractor'],producer=ref(hit['folder']/'producer.json'),silence_padding=False)
        path=out/('receipt_'+str(i)+'.json');save(path,receipt)
        result.append(dict(row,jukebox=hit['features'],jukebox_receipt=ref(path)))
        if i%25==0:save(out/'progress.json',dict(completed=i+1,total=len(manifest['rows']),unique=len(cache)))
    manifest=dict(manifest,stage='complete_features',rows=result)
    save(out/'manifest.json',manifest);prepared=dict(assets,paired_generation_manifest=ref(out/'manifest.json'))
    save(out/'assets.json',prepared)
    _,_,report=inspect(prepared);save(out/'preflight.json',report)
    save(out/'status.json',dict(status='PAIRED_ASSETS_READY' if report['ready'] else 'PAIRED_ASSETS_BLOCKED',
        rows=len(result),unique_audio_windows=len(cache),ready=report['ready'],optimizer_steps=0))
    return out
