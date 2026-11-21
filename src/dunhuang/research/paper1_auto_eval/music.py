"""Auditable automatic music evidence and an explicitly adapted, directional BAS."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from .common import CFG,read,sha,uid
from dunhuang.research.paired_event_r41.events import profiles,phases
from pathlib import Path

STREAMS=[('beat_this','beat'),('beat_this','downbeat'),('allin1','beat'),('allin1','downbeat'),('allin1','section_boundary'),('acoustic','accent_candidate'),('multi_cue','phrase_boundary_candidate'),('R64','beat'),('R64','accent_candidate'),('R64','phrase_proxy'),('R64','section_proxy')]

def catalogs():
    root=Path('outputs/music_events_01/main');assert sha(root/'manifest.json')==CFG['music_main_manifest'];out=[]
    for i,h in enumerate(CFG['music_catalog_sha256']):
        p=root/str(i)/'catalog.json';assert sha(p)==h;c=read(p);audio=Path('outputs/complete_r36_01/viewer/audio')/('dunhuangwu%d.wav'%(i+2));assert sha(audio)==c['audio_sha256'];out.append(c)
    return out

def bind(c,song,interval):
    A,B=map(int,interval)
    if c['fps']!=30 or not 0<=A<B<=c['frames']:raise ValueError('Actual full audio support required')
    events=[]
    for e in c['events']:
        if not e['seconds_supported'] or not A/30<=e['seconds']<B/30:continue
        t=float(e['seconds'])*30-A;nearest=int(np.rint(t))
        events.append(dict(e,local_seconds=t/30,local_frame_float=t,nearest_local_frame=nearest,nearest_local_frame_supported=0<=nearest<B-A))
    return dict(id=uid([c['audio_sha256'],A,B]),audio_sha256=c['audio_sha256'],catalog_sha256=CFG['music_catalog_sha256'][song],model_manifest_sha256=c['model_manifest_sha256'],interval=[A,B],fps=30,events=events,annotation_status='AUTOMATIC_UNVALIDATED',independent_phrase_truth='UNKNOWN',used_for_selection=False)

def motion_beats(P):
    P=np.asarray(P,float)
    if P.ndim!=3 or P.shape[-1]!=3 or len(P)<4 or not np.isfinite(P).all():raise ValueError('Observed finite XYZ sequence required')
    velocity=np.linalg.norm(np.diff(P,axis=0),axis=-1).mean(-1)
    filtered=gaussian_filter1d(velocity,CFG['velocity_smoothing_sigma_frames'])
    return argrelextrema(filtered,np.less)[0]/30

def gaussian_match(query,reference):
    a=np.asarray(query,float);b=np.asarray(reference,float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Finite clock required')
    if not len(a) or not len(b):return None
    return float(np.exp(-np.min((a[:,None]-b[None,:])**2,axis=1)/(2*CFG['alignment_sigma_seconds']**2)).mean())

def distances(query,reference):
    if not len(query) or not len(reference):return dict(mean_abs_distance_seconds=None,coverage_at_0_1_seconds=None)
    d=np.min(abs(np.asarray(query)[:,None]-np.asarray(reference)[None,:]),axis=1)
    return dict(mean_abs_distance_seconds=float(d.mean()),coverage_at_0_1_seconds=float(np.mean(d<=.1)))

def motion_events(R):
    cfg=read('src/dunhuang/research/manifold_bridge_r47/config.json')
    return phases(profiles(R),cfg)

def scores(beats,own,donor,start,end):
    """Same music queries across all non-circular shifts; fixed actual motion."""
    beats=np.asarray(beats,float);margin=CFG['music_query_margin_seconds'];rows=[]
    if not start+margin<end-margin:raise ValueError('Common offset support is empty')
    for source,kind in STREAMS:
        get=lambda c:np.asarray([e['seconds'] for e in c['events'] if e['source']==source and e['kind']==kind and e['seconds_supported']],float)
        a=get(own);a=a[(a>=start+margin)&(a<end-margin)];b=get(donor);b=b[(b>=start+margin)&(b<end-margin)]
        for shift in CFG['clock_shifts_seconds']:
            rows.append(dict(stream=source+':'+kind,condition='assigned' if shift==0 else 'shift_%+.2f'%shift,shift_seconds=shift,score=gaussian_match(a+shift,beats),music_events=len(a),motion_events=len(beats),direction='MUSIC_TO_MOTION',query_support=[start+margin,end-margin],**distances(a+shift,beats)))
        rows.append(dict(stream=source+':'+kind,condition='wrong_song',shift_seconds=None,score=gaussian_match(b,beats),music_events=len(b),motion_events=len(beats),direction='MUSIC_TO_MOTION',query_support=[start+margin,end-margin],**distances(b,beats)))
    return rows
