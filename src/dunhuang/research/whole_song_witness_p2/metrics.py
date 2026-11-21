import numpy as np
from scipy.signal import find_peaks
from dunhuang.research.edit_support_r61.fixed_gap import derivatives
from .common import relative_fk,array_hash

def summary(x):
    x=np.asarray(x,float)
    return dict(count=len(x),mean=float(x.mean()),q95=float(np.quantile(x,.95)),maximum=float(x.max())) if len(x) else dict(count=0,mean=None,q95=None,maximum=None)

def partition(kind,owner,n):
    ids=np.arange(len(kind)-n)
    same=np.all(np.stack([owner[i:i+len(ids)] for i in range(n+1)])==owner[:len(ids)],axis=0)
    stable=np.all(np.stack([kind[i:i+len(ids)] for i in range(n+1)])==kind[:len(ids)],axis=0)&same
    return dict(core_internal=stable&(kind[:len(ids)]==0),bridge_internal=stable&(kind[:len(ids)]==1),cross_region=~stable)

def evaluate(R,J,parents,kind,owner):
    P=relative_fk(R,J,parents);curves={};dynamic={};per_region=[]
    for n in (1,2,3):
        x=np.sqrt(np.mean((np.diff(P,n=n,axis=0)*30**n)**2,axis=(1,2)));curves['D'+str(n)]=x
        masks=partition(kind,owner,n);assert sum(int(v.sum()) for v in masks.values())==len(x)
        dynamic['D'+str(n)]=dict(full=summary(x),**{name:summary(x[mask]) for name,mask in masks.items()})
    for k in (0,1):
        for index in sorted(set(owner[kind==k].tolist())):
            lo=np.flatnonzero((kind==k)&(owner==index));a,b=int(lo[0]),int(lo[-1])+1
            per_region.append(dict(kind='core' if k==0 else 'bridge',index=index,interval=[a,b],internal={name:summary(x[a:max(a,b-int(name[1]))]) for name,x in curves.items()}))
    omega,alpha=derivatives(R)
    result=dict(FK_RMS_derivatives=dynamic,SO3_angular_speed_radps=summary(np.linalg.norm(omega,axis=-1).mean(-1)),
        SO3_transported_acceleration_radps2=summary(np.linalg.norm(alpha,axis=-1).mean(-1)),regions=per_region,
        FK_path_length_mean_m=float(np.linalg.norm(np.diff(P,axis=0),axis=-1).sum(0).mean()),
        FK_range_rms_m=float(np.sqrt(np.mean(np.var(P,axis=0)))),FK_positions_sha256=array_hash(P.astype('<f4')),
        derivative_contract='RMS_XYZ_BODY22_FINITE_DIFFERENCE_AT_30FPS; LEFT_STENCIL_ORIGIN; CROSS_REGION_STENCILS_SEPARATE',
        scope='ROOT_RELATIVE_BODY22_NOT_NATIVE24_OR_PERCEPTUAL_QUALITY')
    return P,curves,result

def music_diagnostic(P,catalog):
    """Actual output re-detection, no mapped old peaks and no quality claim."""
    activity=np.linalg.norm(np.diff(P,axis=0,prepend=P[:1])*30,axis=-1).mean(-1)
    peaks=find_peaks(activity,distance=8,prominence=max(float(np.ptp(activity))*.1,1e-8))[0]
    evidence={}
    for source,kind in sorted({(e['source'],e['kind']) for e in catalog['events']}):
        events=np.array([e['frame_float'] for e in catalog['events'] if e['source']==source and e['kind']==kind],float)
        controls={}
        # Non-circular shifts, identical common support across all five clocks.
        for shift in (0,-15,-8,8,15):
            p=peaks[(peaks>=15)&(peaks<len(P)-15)];t=events[(events>=15)&(events<len(P)-15)]+shift
            d=abs(p[:,None]-t[None]).min(1) if len(p) and len(t) else np.array([])
            controls[str(shift)]=dict(mean_nearest_frames=float(d.mean()) if len(d) else None,gaussian_sigma3=float(np.exp(-d*d/18).mean()) if len(d) else None,motion_events=len(p),music_events=len(t))
        evidence[source+':'+kind]=controls
    return dict(actual_output_motion_peaks=peaks.tolist(),event_type_and_model_separate=evidence,truth='AUTOMATIC_UNVALIDATED_EVENTS; NO_ARTISTIC_OR_CLOCK_ACCURACY_CLAIM')
