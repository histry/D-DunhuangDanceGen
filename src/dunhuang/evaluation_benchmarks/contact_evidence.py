"""External contact observations: never infer intentional slide from speed."""
import json
import numpy as np
from dunhuang.io import checked

FEET=(7,8,10,11)


def distribution(values,possible,units):
    x=np.asarray(values,dtype=float).reshape(-1)
    if not np.isfinite(x).all():raise ValueError('Nonfinite measurements')
    return dict(status='COMPUTED' if len(x) else 'NOT_EVALUABLE',
        sample_count=len(x),possible_samples=int(possible),
        coverage=float(len(x)/possible) if possible else 0.,units=units,
        mean=float(x.mean()) if len(x) else None,
        p95=float(np.percentile(x,95)) if len(x) else None,
        max=float(x.max()) if len(x) else None)


def load_evidence(reference,motion_sha256,frames):
    if reference is None:return None,None
    value=json.loads(checked(reference).read_text())
    if (value.get('schema')!='independent_contact_labels_v1' or value.get('validated') is not True or
        value.get('motion_sha256')!=motion_sha256 or value.get('fps')!=30 or value.get('frames')!=frames or
        value.get('foot_joint_indices')!=list(FEET)):
        raise ValueError('Contact labels lack validated motion/clock identity')
    if (value.get('source') not in ('manual_video_annotation','force_plate') or
        value.get('independent_of_evaluation_metric') is not True or not value.get('reviewer')):
        raise ValueError('Contact semantics need independent evidence')
    labels=np.load(checked(value['labels']),allow_pickle=False)
    if labels.dtype!=np.int8 or labels.shape!=(frames,4) or not np.isin(labels,[-1,0,1,2]).all():
        raise ValueError('External contact label population differs')
    return labels,value


def conditioned_metrics(joints,floor_y,labels=None,derived=None,low=None,predicted=None):
    x=np.asarray(joints,dtype=np.float32)
    if x.ndim!=3 or x.shape[1:]!=(24,3) or len(x)<4 or not np.isfinite(x).all() or not np.isfinite(floor_y):
        raise ValueError('Finite 30FPS matching SMPL24 FK and floor required')
    feet=x[:,FEET];n=len(x);shape=(n,4)
    def mask(value):return np.zeros(shape,bool) if value is None else np.asarray(value,bool)
    d,h,p=mask(derived),mask(low),mask(predicted)
    if any(v.shape!=shape for v in (d,h,p)):raise ValueError('Four-foot evidence masks differ')
    candidate=d|h
    state=np.full(shape,-1,np.int8)
    # Lack of support evidence is not a verified swing/flight label either.
    if labels is not None:
        a=np.asarray(labels)
        if a.shape!=shape or not np.isin(a,[-1,0,1,2]).all():raise ValueError('Contact-state population differs')
        state=a.astype(np.int8)
    delta=np.diff(feet[:,:,[0,2]],axis=0)
    speed=np.linalg.norm(delta,axis=-1)*30
    def population(k):
        pairs=(state[:-1]==k)&(state[1:]==k)
        return distribution(speed[pairs],speed.size,'m/s')
    result=dict(schema='contact_evidence_conditioned_v2',semantic_status='UNVERIFIED_ALL_UNKNOWN' if labels is None else 'EXTERNAL_LABELS',
        foot_joint_indices=list(FEET),contact_type_counts={str(k):int((state==k).sum()) for k in (-1,0,1,2)},
        unknown_fraction=float((state==-1).mean()),support_observation_frames=int(candidate.sum()),
        support_coverage=float(candidate.mean()),independent_static_speed=population(1),
        independent_sliding_speed=population(2),
        intentional_sliding_inferred_from_speed=False,valid_static_drift=None,
        evidence_counts=dict(derived_only=int((d&~h).sum()),height_only=int((h&~d).sum()),
                            both=int((d&h).sum()),predicted_contact=int(p.sum()),
                            derived_contact=int(d.sum()),height_observation=int(h.sum()),
                            prediction_derived_disagreement=int((p!=d).sum())),
        units=dict(speed='m/s',height='m',drift='m'),floor_y_m=float(floor_y))
    drifts=[]
    for f in range(4):
        active=state[:,f]==1
        starts=np.flatnonzero(np.diff(np.r_[False,active,False].astype(int))==1)
        ends=np.flatnonzero(np.diff(np.r_[False,active,False].astype(int))==-1)
        for a,b in zip(starts,ends):
            if b-a>=2:drifts.extend(np.linalg.norm(feet[a:b,f][:,[0,2]]-feet[a,f,[0,2]],axis=-1))
    result['valid_static_drift']=distribution(drifts,n*4,'m')
    # Geometry-only populations stay separate, never relabelled as static truth.
    for name,m in [('derived',d),('height_only',h&~d),('union',candidate)]:
        pairs=m[:-1]&m[1:]
        result[name+'_observation_speed']=distribution(speed[pairs],speed.size,'m/s')
    return result
