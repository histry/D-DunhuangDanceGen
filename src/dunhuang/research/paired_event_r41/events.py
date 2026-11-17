"""Whole-record kinematic candidates, with protected holds and explicit unknown semantics."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

REGIONS={'upper':(12,13,14,15,16,17,18,19,20,21),'lower':(1,2,4,5,7,8,10,11),'trunk':(0,3,6,9)}


def profiles(rotations,root=None):
    r=np.asarray(rotations,float)
    if r.ndim!=4 or r.shape[1:]!=(22,3,3) or len(r)<4 or not np.isfinite(r).all():raise ValueError('Real full-record common22 rotations')
    omega=np.empty((len(r),22,3));omega[1:]=Rotation.from_matrix((r[:-1].swapaxes(-1,-2)@r[1:]).reshape(-1,3,3)).as_rotvec().reshape(-1,22,3)*30;omega[0]=omega[1]
    speed=np.linalg.norm(omega,axis=-1);result={k:speed[:,v].mean(-1) for k,v in REGIONS.items()};result['global']=speed.mean(-1);result['omega']=omega
    v=omega.reshape(len(r),-1);norm=np.linalg.norm(v,axis=-1);cos=(v[1:]*v[:-1]).sum(-1)/np.maximum(norm[1:]*norm[:-1],1e-12);result['direction_change']=np.r_[0,1-np.clip(cos,-1,1)];result['root_angular_speed']=speed[:,0]
    if root is not None:
        root=np.asarray(root,float)
        if root.shape!=(len(r),3) or not np.isfinite(root).all():raise ValueError('Observed metric relative Root')
        result['root_speed']=np.r_[0,np.linalg.norm(np.diff(root,axis=0)*30,axis=-1)]
    return result


def spans(mask,min_length):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return [(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=min_length]


def phases(p,cfg):
    out=[]
    for region in REGIONS:
        speed=p[region];scale=max(float(np.median(abs(speed-np.median(speed)))*1.4826),.05)
        for kind,signal,floor in [('speed_peak',speed,.15),('motion_onset',np.maximum(np.diff(speed,prepend=speed[0])*30,0),1.),('deceleration',np.maximum(-np.diff(speed,prepend=speed[0])*30,0),1.)]:
            ids,properties=find_peaks(signal,distance=cfg['peak_min_distance'],prominence=max(scale,floor))
            for t,v in zip(ids,properties['prominences']):out.append(dict(frame=int(t),type=kind,body_region=region,score=float(v/(v+scale+floor)),score_scope='UNCALIBRATED_KINEMATIC_PROXY',evidence_span=[max(0,int(t)-1),min(len(speed),int(t)+2)],semantic_truth='UNKNOWN'))
    for a,b in spans(p['root_angular_speed']>max(.8,float(np.quantile(p['root_angular_speed'],.8))),6):out.append(dict(frame=(a+b-1)//2,type='root_turn_span',body_region='root',score=None,evidence_span=[a,b],semantic_truth='UNKNOWN'))
    speed=p['global'];threshold=min(float(np.quantile(speed,.2)),cfg['hold_max_speed_radps'])
    for a,b in spans(speed<=threshold,cfg['hold_min_frames']):out.append(dict(frame=(a+b-1)//2,type='pose_hold',body_region='global',score=None,evidence_span=[a,b],semantic_truth='UNKNOWN'))
    ids,_=find_peaks(np.where(speed>.2,p['direction_change'],0),distance=cfg['peak_min_distance'],prominence=.3)
    for t in ids:out.append(dict(frame=int(t),type='direction_turn',body_region='global',score=None,evidence_span=[max(0,int(t)-3),min(len(speed),int(t)+4)],semantic_truth='UNKNOWN'))
    return sorted(out,key=lambda x:(x['frame'],x['body_region'],x['type']))


def segment(p,phase,cfg,interval=None):
    n=len(p['global']);lo,hi=(0,n) if interval is None else map(int,interval)
    if not 0<=lo<hi<=n:raise ValueError('True source support')
    protected=[q['evidence_span'] for q in phase if q['type'] in ('pose_hold','root_turn_span')]
    # No absolute speed threshold copied from ChangE upper8: regions contribute after robust scaling.
    signals=[]
    for name in ('upper','lower','trunk','root_speed'):
        if name in p:
            v=gaussian_filter1d(p[name],cfg['boundary_smoothing_sigma_frames']);signals.append(v/max(float(np.quantile(v,.75)),.05))
    activity=np.mean(signals,axis=0);minimum=cfg['minimum_core_frames'];maximum=cfg['maximum_core_frames'];preferred=cfg['preferred_core_frames']
    valleys=find_peaks(-activity,distance=cfg['boundary_grid_frames'])[0]
    candidates=sorted(set([lo,hi]+list(range(lo+minimum,hi-minimum+1,cfg['boundary_grid_frames']))+[int(i) for i in valleys if lo+minimum<=i<=hi-minimum]))
    candidates=[i for i in candidates if i in (lo,hi) or not any(a<i<b for a,b in protected)]
    # A full protected hold is never cut to satisfy a hard length cap.
    states={lo:(0.,[])}
    for b in candidates[1:]:
        choices=[]
        for a,(cost,path) in states.items():
            length=b-a
            if length<minimum and not (a==lo and b==hi):continue
            if length>maximum:
                intervening=[t for t in candidates if a+minimum<=t<=b-minimum]
                if intervening:continue
            edge=0. if b==hi else float(activity[b]);value=cost+edge+.4*((length-preferred)/preferred)**2+.08
            choices.append((value,path+[[a,b]]))
        if choices:states[b]=min(choices,key=lambda z:(z[0],z[1]))
    if hi not in states:return [[lo,hi]],dict(fallback='UNSPLIT_SOURCE_NO_VALID_PATH',protected_spans=protected)
    return states[hi][1],dict(fallback=None,protected_spans=protected,boundary_cost=states[hi][0])


def clock_relations(rows,positive_iou=.6,negative_gap=15):
    n=len(rows);positive=np.zeros((n,n),bool);negative=positive.copy()
    for i,a in enumerate(rows):
        for j,b in enumerate(rows):
            if a['role']!=b['role']:continue
            A=a['audio_frame_interval'];B=b['audio_frame_interval'];intersection=max(0,min(A[1],B[1])-max(A[0],B[0]));union=max(A[1],B[1])-min(A[0],B[0]);same=a['audio_sha256']==b['audio_sha256']
            if same and intersection/max(union,1e-12)>=positive_iou:positive[i,j]=True
            elif not same or max(A[0],B[0])-min(A[1],B[1])>=negative_gap:negative[i,j]=True
    return positive,negative


def validate_roles(rows):
    maps={k:{} for k in ('raw_motion_sha256','audio_sha256','source_group')}
    for row in rows:
        if row['role'] not in ('train','val'):raise ValueError('Test tensors forbidden')
        for key,m in maps.items():
            if m.setdefault(row[key],row['role'])!=row['role']:raise ValueError(key+' crosses roles')
    return {key:len(value) for key,value in maps.items()}
