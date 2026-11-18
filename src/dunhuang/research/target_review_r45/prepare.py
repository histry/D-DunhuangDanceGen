"""One metadata-selected candidate/source/query; separate original context and music-timed motion."""
import argparse,base64,gzip,hashlib,json,shutil,datetime
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.io import checked
from dunhuang.research.complete_r36.core import read,write,sha,Library,relative_fk
from dunhuang.research.reperformance_r44.archive import Arrays

def hash_bytes(x):return hashlib.sha256(x).hexdigest()

def sample_bounds(interval,sample_rate,samples):
    A,B=map(int,interval)
    if not 0<=A<B:raise ValueError('Actual half-open pose-frame interval required')
    start=round(A*sample_rate/30);requested=round(B*sample_rate/30);last_center=round((B-1)*sample_rate/30)
    if last_center>=samples or start>=samples:raise ValueError('Registered final pose centre has no real audio support')
    finish=min(requested,samples)
    return start,finish,dict(requested_sample_interval=[start,requested],actual_sample_interval=[start,finish],last_pose_center_sample=last_center,last_pose_center_observed=True,partial_final_audio_bin=finish<requested,policy='REAL_EOF_WITH_SUPPORTED_FINAL_FRAME_CENTER; NO_ZERO_PADDING_OR_SLOT_CHANGE')

def prepare(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh blinded package')
    view=out/'reviewer';view.mkdir(parents=True);admin=out/'admin';admin.mkdir();parent=Path('outputs/retrieval_audit_r44_01/native');native=Path('outputs/sdmr_event_rank_01/change_shadow_v3');meta=read(native/'result.json');old=read(parent/'admin_map.json');lookup={(r['key'],r['candidate']):r for r in old};z=Arrays(native/'arrays.npz');J=z['skeleton_J22'];parents=z['parents'];lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog']);tracks={};queries=[];items=[];selection=[];verified_audio=set();arrays={}
    def track(x):
        key=str(len(tracks));a=np.asarray(x,dtype='<f4');tracks[key]=dict(shape=list(a.shape),data=base64.b64encode(a.tobytes()).decode());return key,hash_bytes(a.tobytes())
    for row in meta['records']:
        key=row['key'];qid=hash_bytes(('R45_QUERY:'+key).encode())[:20];source=checked(row['query_audio']) if row['query_audio']['sha256'] not in verified_audio else Path(row['query_audio']['path']);verified_audio.add(row['query_audio']['sha256']);A,B=row['audio_frame_interval']
        with sf.SoundFile(source) as f:
            start,finish,audio_clock=sample_bounds([A,B],f.samplerate,len(f));f.seek(start);wave=f.read(finish-start,dtype='float32',always_2d=True);sample_rate=f.samplerate
        audio=qid+'.wav';sf.write(view/audio,wave,sample_rate,subtype='PCM_16');audio_sha=sha(view/audio);candidates=[]
        for uid in sorted({c['source_uid'] for c in row['candidates']}):
            options=[j for j,c in enumerate(row['candidates']) if c['source_uid']==uid];j=min(options,key=lambda j:hash_bytes(('R45_BALANCE:'+lookup[key,j]['blind_item_id']).encode()));c=row['candidates'][j];bid=lookup[key,j]['blind_item_id'];a,b=map(int,c['source_interval']);lo=max(0,a-30);hi=min(len(lib.rotations[uid]),b+30);context=relative_fk(lib.rotations[uid][lo:hi],J,parents);timed=z[key+f'_c{j}_xyz'];ct,ch=track(context);tt,th=track(timed);arrays[bid+'_original_context']=context;arrays[bid+'_retimed']=timed
            original_ref=hash_bytes((ch+json.dumps([a-lo,b-lo])+hash_bytes(parents.tobytes())).encode());match_ref=hash_bytes((th+audio_sha+hash_bytes(parents.tobytes())).encode());candidates.append(dict(id=bid,context_track=ct,retimed_track=tt,context_frames=len(context),target_frames=len(timed),candidate_support=[a-lo,b-lo],media_ref=dict(EVENT=original_ref,POSE=original_ref,MUSIC_MATCH=match_ref)))
            items.append(dict(blind_item_id=bid,query_id=qid,key=key,candidate=j,source_uid=uid,recording_uid=c['recording_uid'],raw_motion_sha256=lib.rows[uid][0]['source_sha256'],native_motion_sha256=c['reference']['sha256'],raw_motion_identity_scope='SEALED_CATALOG_SOURCE_IDENTITY',source_interval=[a,b],context_interval=[lo,hi],event_uid=c['event_uid'],query_audio=row['query_audio'],audio_frame_interval=[A,B],audio_sample_interval=[start,finish],audio_clock=audio_clock,audio_crop_sha256=audio_sha,original_xyz_sha256=ch,retimed_xyz_sha256=th,media_ref=candidates[-1]['media_ref'],reference=c['reference'],time_warp=c['time_warp'],scope='KINEMATIC_SOURCE_REVIEW_NOT_ASSUMED_SEMANTIC_TRUTH'));selection.append(dict(key=key,candidate=j,blind_item_id=bid))
        candidates.sort(key=lambda c:hash_bytes(('R45_PRESENT:'+c['id']).encode()));query_ref=hash_bytes((audio_sha+''.join(sorted(c['id'] for c in candidates))).encode());queries.append(dict(id=qid,audio=audio,frames=row['frames'],candidates=candidates,query_media_ref=query_ref))
    payload=dict(queries=queries,tracks=tracks,parents=parents.tolist(),fps=30,protocol='DUNHUANG_BLIND_REVIEW_V1_R45',status='PREPARATION_ONLY',human_reviewed=False,scope='E/P original30Hz source with context; M actual E1-W target clock. Common22 Root-relative, no finger/world/contact truth.');(view/'data.js').write_text('window.R45_REVIEW="'+base64.b64encode(gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode(),mtime=0)).decode()+'";',encoding='utf8')
    for f in ('index.html','review.js','style.css'):shutil.copyfile(Path(__file__).with_name(f),view/f)
    np.savez_compressed(admin/'arrays.npz',**arrays);write(admin/'manifest.json',dict(items=items,selection=selection,queries=[dict(query_id=q['id'],query_media_ref=q['query_media_ref']) for q in queries],native_manifest_sha256=sha(native/'result.json'),old_review_map_sha256=sha(parent/'admin_map.json'),original_pool=552,selected=len(items),queries_count=len(queries),rule='ONE_PER_RAW_SOURCE_QUERY_MIN_SHA256_BLIND_ID; SCORES_NOT_READ',role='HISTORICALLY_EXPOSED_DUNHUANG_TRANSFER',independent_new_music_count=0,status='PREPARATION_ONLY',human_reviewed=False,can_route=False));write(view/'protocol.json',dict(protocol=payload['protocol'],items=len(items),queries=len(queries),status='PREPARATION_ONLY',human_reviewed=False,files=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(view.iterdir()) if f.name!='protocol.json']));z.close();print('R45_BALANCED_REVIEW',len(items),len(queries),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();prepare(a.out)
