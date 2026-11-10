"""Run fresh 5s, 15s, whole-song baseline and matched sparse-anchor experiments."""
import json
import sys
import traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import pinned, sha, write, checked
from dunhuang.pipeline import run
from dunhuang.retrieval import retrieve
from dunhuang.contracts import MotionAdapter

root=Path(__file__).resolve().parents[1]
q=json.loads((root/'configs/generation.json').read_text())
out=root/'outputs/experiment_v1'
if out.exists() and any(out.iterdir()):raise ValueError('Existing experiment outputs must not be overwritten')
out.mkdir(parents=True,exist_ok=True)
legacy=pinned(q['legacy_root'],q['legacy_commit']);sys.path.insert(0,str(legacy))
from motion.retrieval_generation import decode_continuous_source
from motion.geometry import fk_24_np
catalog_path=legacy/'outputs/unpaired_data_protocol_20261015_ZY1C32/catalog/motion_catalog.json'
catalog=json.loads(catalog_path.read_text());entries=[]
# Predeclared deterministic coverage: one first event from each of three source recordings.
for source in sorted(catalog['source_records'],key=lambda r:r['source_uid'])[:3]:
    event=min((e for e in catalog['events'] if e['source_uid']==source['source_uid']),key=lambda e:e['core_frame_start'])
    raw=Path(q['asset_root'])/'assets/motion/smpl_official_14'/Path(source['raw_receipt']['path']).name
    checked(dict(path=str(raw),sha256=source['source_sha256']))
    a,b=event['core_frame_start'],event['core_frame_end']
    motion,_=decode_continuous_source(raw,source['discovery_metadata'],(a,b),known_rows=np.ones(b-a,bool))
    recovered=MotionAdapter.to_native(MotionAdapter.to_edge(motion))
    error=float(abs(fk_24_np(recovered)-fk_24_np(motion)).max())
    if error>1e-5:raise ValueError('Real-source adapter FK roundtrip failed')
    target=out/(source['source_uid']+'.npy');np.save(target,motion)
    entries.append(dict(id=event['event_uid'],source_uid=source['source_uid'],source_frame=len(motion)//2,
        motion=dict(path=str(target),sha256=sha(target)),source_sha256=source['source_sha256'],
        raw_interval=[a,b],fk_roundtrip_max_abs_m=error))
write(out/'catalog.json',dict(entries=entries,source_catalog_sha256=sha(catalog_path),selection='first event, first three sorted sources; no best-of or outcome selection'))
selected=retrieve(dict(entries=entries),limit=3)
# Acoustic beats from the actual query WAV, independent of generated motion.
import librosa
audio,sr=librosa.load(checked(q['audio']),sr=22050)
_,beat_times=librosa.beat.beat_track(y=audio,sr=sr,units='time')
beats=np.unique(np.rint(np.asarray(beat_times)*30).astype(int)).tolist()
write(out/'timeline.json',dict(beat_frames=beats,fps=30,method='librosa beat_track',audio=q['audio']))
results=[]
for frames,label in [(150,'5s'),(450,'15s'),(3601,'whole')]:
    anchors=[]
    for index,t in enumerate(np.linspace(frames*.2,frames*.8,3)):
        valid=[b for b in beats if 0<b<frames-1]
        frame=min(valid,key=lambda b:abs(b-t)) if valid else int(t)
        row=selected[index]
        anchors.append(dict(frame=frame,pose=row['pose'].tolist(),joints=list(range(12,22)),
            catalog_id=row['id'],source_frame=row['source_frame'],source=row['source']))
    for mode in ['baseline','anchored']:
        case=dict(q,frames=frames,mode=mode,output=str(out/(mode+'_'+label)),anchors=anchors if mode=='anchored' else [],beat_frames=[b for b in beats if b<frames])
        # Evaluate the baseline against the exact same anchor targets after generation.
        print('START',mode,label,flush=True)
        try:
            path=run(case)
            from dunhuang.evaluation import evaluate
            write(path/'matched_evaluation.json',evaluate(np.load(path/'native.npy'),np.load(path/'native_fk.npy'),anchors,case['beat_frames']))
            results.append(dict(case=mode+'_'+label,**json.loads((path/'status.json').read_text())))
        except Exception:
            results.append(dict(case=mode+'_'+label,status='FAILED',error=traceback.format_exc()))
        write(out/'summary.json',dict(cases=results,training=False,independent_style_verified=False,full_generation_qualified=False))
        print('FINISH',results[-1],flush=True)
if any(c['status']!='COMPLETED' for c in results):raise SystemExit(1)
