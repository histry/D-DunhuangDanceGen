"""Same-anchor factor control and a fixed source-only feasible clock bank."""
import numpy as np
from scipy.interpolate import BPoly
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.research.reperformance_r44.phase import UPPER,smooth_map,certificate
from dunhuang.research.reperformance_r44.motion import decode,retime


def snap(x):
    x=np.asarray(x,float).copy();close=abs(x-np.rint(x))<1e-10;x[close]=np.rint(x[close]);return x


def matched_maps(records,n=256):
    linear=np.arange(n,dtype=float);smooth=linear.copy()
    for r in records:
        a,b=r['interval'];s=np.asarray(r['source']);t=np.asarray(r['target']);p=BPoly.from_derivatives(t,[[x,1.,0.] for x in s],extrapolate=False);linear[a:b]=np.interp(np.arange(a,b),t,s);smooth[a:b]=p(np.arange(a,b))
    return snap(linear),snap(smooth)


def stages(raw,J,parents,a,b,cfg):
    R,_=decode(raw);lo,hi=(0,104) if b<=104 else (152,256);part=R[lo:hi];v=Rotation.from_matrix((part[:-1].swapaxes(-1,-2)@part[1:]).reshape(-1,3,3)).magnitude().reshape(-1,22)[:,UPPER].mean(-1)*30;speed=gaussian_filter1d(np.r_[v[0],v],1);acc=np.gradient(speed);candidates=[]
    if speed.max()<cfg['minimum_motion_speed_radps']:return [],dict(status='LOW_ACTIVITY_NO_FORCED_PHASE',maximum_speed=float(speed.max()))
    def peaks(values,kind,threshold):
        ids,_=find_peaks(values,distance=cfg['minimum_stage_separation_frames'],prominence=max(threshold,.15*float(np.std(values))))
        for i in ids:
            if a+10<lo+i<b-11:candidates.append(dict(frame=int(lo+i),type=kind,salience=float(values[i]),semantic_truth='UNKNOWN',observed=True))
    peaks(speed,'speed_peak',.05);peaks(np.maximum(acc,0),'motion_onset_proxy',.015);peaks(np.maximum(-acc,0),'braking_proxy',.015)
    P=relative_fk(part,J,parents);vel=np.diff(P[:,[20,21]],axis=0);norm=np.linalg.norm(vel,axis=-1);cos=np.sum(vel[1:]*vel[:-1],axis=-1)/np.maximum(norm[1:]*norm[:-1],1e-12);turn=np.r_[0.,0.,np.mean(np.where((norm[1:]>.001)&(norm[:-1]>.001),np.arccos(np.clip(cos,-1,1)),0),axis=1)];peaks(turn,'direction_turn_proxy',.2)
    low=speed<max(.05,.12*float(np.quantile(speed,.95)));edges=np.diff(np.r_[False,low,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    for x,y in zip(starts,ends):
        frame=lo+(x+y-1)//2
        if y-x>=cfg['minimum_hold_frames'] and a+10<frame<b-11:candidates.append(dict(frame=int(frame),type='pose_hold_proxy',interval=[int(lo+x),int(lo+y)],salience=float((y-x)/len(speed)),semantic_truth='UNKNOWN',observed=True))
    # A type-diverse, fixed metadata/salience rule, not a model or music-dependent choice.
    selected=[]
    for kind in ('speed_peak','motion_onset_proxy','braking_proxy','direction_turn_proxy','pose_hold_proxy'):
        for p in sorted([p for p in candidates if p['type']==kind],key=lambda p:(-p['salience'],p['frame'])):
            if all(abs(p['frame']-q['frame'])>=cfg['minimum_stage_separation_frames'] for q in selected):selected.append(p);break
        if len(selected)==cfg['maximum_stages_per_core']:break
    return sorted(selected,key=lambda p:p['frame']),dict(status='KINEMATIC_PROXY_NOT_ARTISTIC_PHASE',candidates=candidates,visible_derivative_support=[lo,hi],maximum_speed=float(speed.max()))


def patterns(k):
    result=[np.zeros(k)]
    if k:
        result.extend(np.full(k,x) for x in (-6,-3,3,6));result.extend([np.linspace(-6,6,k),np.linspace(6,-6,k),np.array([3*(-1)**i for i in range(k)]),np.array([-3*(-1)**i for i in range(k)])])
        for i in range(k):
            for x in (-6,-3,3,6):p=np.zeros(k);p[i]=x;result.append(p)
    return result


def part_metrics(raw,J,parents,a,b):
    P=relative_fk(decode(raw)[0],J,parents)[a:b]
    return {f'D{n}':float(np.sqrt(np.mean((np.diff(P,n=n,axis=0)*30**n)**2))) for n in (1,2,3)}


def feasible_bank(raw,J,parents,a,b,cfg):
    phases,evidence=stages(raw,J,parents,a,b,cfg);source=np.array([a]+[p['frame'] for p in phases]+[b-1],float);baseline=part_metrics(raw,J,parents,a,b);bank=[];seen=set()
    for number,pattern in enumerate(patterns(len(phases))[:cfg['maximum_clock_patterns_per_core']]):
        p,r=smooth_map(source,source+np.r_[0,pattern,0],cfg);signature=tuple(r['target'])
        if signature in seen:continue
        seen.add(signature);q=np.arange(256,dtype=float);q[a:b]=p(np.arange(a,b));q=snap(q);out=retime(raw,q);m=part_metrics(out,J,parents,a,b);ratio={k:m[k]/max(baseline[k],1e-12) if baseline[k]>1e-12 else (1. if m[k]<1e-12 else None) for k in m};eligible=ratio['D1'] is not None and cfg['activity_ratio'][0]<=ratio['D1']<=cfg['activity_ratio'][1] and all(ratio[f'D{n}'] is not None and ratio[f'D{n}']<=cfg[f'maximum_D{n}_ratio'] for n in (2,3))
        for s,t in zip(source,r['target']):assert np.array_equal(out[int(t),4:],raw[int(s),4:])
        bank.append(dict(pattern_id=number,source=source.tolist(),target=r['target'],certificate=r['certificate'],scale=r['scale'],metrics=m,ratios=ratio,eligible=eligible,q=q,identity=bool(np.array_equal(q,np.arange(256))),score=None))
    assert any(r['identity'] and r['eligible'] for r in bank)
    return bank,phases,evidence


def choose(bank,phases,signal,cfg):
    for r in bank:
        target=np.array(r['target'])[1:-1];s=np.array(r['source'])[1:-1];scores=[]
        for t,p in zip(target,phases):
            if signal is None or not signal['valid']:v=0.
            else:
                i=int(t);on=signal['onset'][i];beat=signal['beat_strength'][i];kind=p['type'];v=.7*on+.3*beat if kind=='speed_peak' else .9*on+.1*beat if kind=='motion_onset_proxy' else 1-on if kind in ('braking_proxy','pose_hold_proxy') else .4*on+.6*beat
            scores.append(float(v))
        displacement=float(np.mean(((target-s)/6)**2)) if len(s) else 0.;strain=float(np.mean((np.diff(r['q'])-1)**2));r['score']=(float(np.mean(scores)) if scores else 0)-cfg['identity_penalty']*displacement-cfg['strain_penalty']*strain
    return max([r for r in bank if r['eligible']],key=lambda r:(r['score'],-r['pattern_id']))


def compile_clock(raw,J,parents,signal,cfg,banks=None):
    q=np.arange(256,dtype=float);records=[];compiled=[]
    for index,(a,b) in enumerate(cfg['core_intervals']):
        bank,phases,evidence=feasible_bank(raw,J,parents,a,b,cfg) if banks is None else banks[index];chosen=choose(bank,phases,signal,cfg);q[a:b]=chosen['q'][a:b];records.append(dict(interval=[a,b],source=chosen['source'],target=chosen['target'],certificate=chosen['certificate'],pattern_id=chosen['pattern_id'],identity=chosen['identity'],score=chosen['score'],phases=phases,evidence=evidence,bank=[{k:v for k,v in r.items() if k!='q'} for r in bank],eligibility_scope='SOURCE_ONLY_ENGINEERING_FILTER_NOT_INDEPENDENT_PHYSICS_EVIDENCE'));compiled.append((bank,phases,evidence))
    return snap(q),records,compiled
