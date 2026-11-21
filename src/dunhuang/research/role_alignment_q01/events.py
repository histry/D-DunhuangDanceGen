"""Whole-record observed proxies, never require a universal stage triple."""
import numpy as np
from .common import CFG,read
from dunhuang.research.paired_event_r41.events import profiles,phases
from dunhuang.research.warp_coupling_r53.common import identity,array_hash

def motion_catalog(R,source_uid,offset=0):
    cfg=read('src/dunhuang/research/manifold_bridge_r47/config.json');rows=[];h=array_hash(R)
    for p in phases(profiles(R),cfg):
        role=None;frame=p['frame']
        if p['body_region']=='upper' and p['type']=='motion_onset':role='ACCEL_RISE'
        elif p['body_region']=='upper' and p['type']=='speed_peak':role='SPEED_PEAK'
        elif p['type']=='direction_turn':role='DIRECTION_TURN'
        elif p['type']=='pose_hold':role='HOLD_END';frame=p['evidence_span'][1]
        if role is None:continue
        supported=bool(0<=frame<len(R) and 0<=p['evidence_span'][0]<p['evidence_span'][1]<=len(R))
        rows.append(dict(uid=identity([source_uid,h,role,frame,p['evidence_span']]),source_uid=source_uid,role=role,frame=int(frame),time=float(frame/30+offset),evidence_span=p['evidence_span'],supported=supported,semantic_role='UNKNOWN',detector_definition='POSITIVE_ANGULAR_SPEED_CHANGE_PEAK' if role=='ACCEL_RISE' else 'R47_WHOLE_RECORD_KINEMATIC_PROXY'))
    return dict(source_uid=source_uid,motion_sha256=h,full_frames=len(R),events=sorted(rows,key=lambda r:(r['frame'],r['role'],r['uid'])),semantic_truth='UNKNOWN')

def select_nodes(events,interval):
    a,b=interval;eligible=[e for e in events if e['supported'] and a+4<e['frame']<b-5 and a<=e['evidence_span'][0]<e['evidence_span'][1]<=b];chosen=[]
    for f in [.2,.5,.8]:
        pool=[e for e in eligible if all(abs(e['frame']-v['frame'])>=12 for v in chosen)]
        if pool:chosen.append(min(pool,key=lambda e:(abs(e['frame']-(a+f*(b-a-1))),e['frame'],e['role'],e['uid'])))
    return sorted(chosen,key=lambda e:e['frame'])

def music_catalog(cat):
    roles={('beat_this','beat'):'BEAT',('beat_this','downbeat'):'DOWNBEAT',('acoustic','accent_candidate'):'ACCENT_PROXY',('acoustic','chroma_boundary_candidate'):'STRUCTURE_PROXY'};rows=[]
    for e in cat['events']:
        key=(e['source'],e['kind'])
        if key not in roles or not e['seconds_supported']:continue
        rows.append(dict(uid=e.get('uid') or identity([cat['audio_sha256'],key,e['seconds']]),role=roles[key],time=float(e['seconds']),source=e['source'],kind=e['kind'],semantic_role='UNKNOWN'))
    return sorted(rows,key=lambda e:(e['time'],e['uid']))
