"""Automatic event observations retain exact seconds and explicit qualification."""
import numpy as np

def event(seconds,kind,source,samples,sr,index,**extra):
    t=float(seconds)
    if not np.isfinite(t):raise ValueError('Nonfinite event time')
    n=(samples*30+sr-1)//sr;frame=int(np.rint(t*30))
    return dict(uid=f'{source}:{kind}:{index}',kind=kind,source=source,seconds=t,
        audio_sample_position=t*sr,frame_float=t*30,nearest_frame=frame,
        seconds_supported=0<=t<samples/sr,nearest_frame_supported=0<=frame<n,
        annotation_status='AUTOMATIC_UNVALIDATED',semantic_truth='UNKNOWN',**extra)

def agreement(a,b,tolerance=.07):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (np.diff(a)<=0).any() or (np.diff(b)<=0).any():raise ValueError('Sorted unique finite seconds required')
    i=j=0;pairs=[]
    while i<len(a) and j<len(b):
        if abs(a[i]-b[j])<=tolerance:pairs.append((i,j,float(b[j]-a[i])));i+=1;j+=1
        elif a[i]<b[j]:i+=1
        else:j+=1
    return dict(a_count=len(a),b_count=len(b),one_to_one_matches=len(pairs),tolerance_seconds=tolerance,
        matched_fraction_a=len(pairs)/len(a) if len(a) else None,matched_fraction_b=len(pairs)/len(b) if len(b) else None,
        symmetric_matched_fraction=2*len(pairs)/(len(a)+len(b)) if len(a)+len(b) else None,
        mean_abs_error_seconds=float(np.mean([abs(p[2]) for p in pairs])) if pairs else None,
        unmatched_a=[i for i in range(len(a)) if i not in {p[0] for p in pairs}],unmatched_b=[j for j in range(len(b)) if j not in {p[1] for p in pairs}],
        pairs=pairs,interpretation='MODEL_AGREEMENT_NOT_ACCURACY')

def scale(x):
    x=np.maximum(np.asarray(x,float),0);q=np.quantile(x,.95)
    return np.clip(x/max(q,1e-12),0,1)

def reward(motion_seconds,audio_events,shift=0.,sigma=.1):
    p=np.asarray(motion_seconds,float)+shift;e=np.asarray(audio_events,float)
    return float(np.exp(-np.min(abs(p[:,None]-e[None,:]),axis=1)**2/(2*sigma*sigma)).mean()) if len(p) and len(e) else None
