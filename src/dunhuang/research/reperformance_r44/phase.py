"""Integer phase anchors, analytical continuous rate certification, exact protected clock."""
import numpy as np
from scipy.interpolate import BPoly,PPoly
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import multi_warp

UPPER=(3,6,9,12,13,14,15,16,17,18,19,20,21)

def certificate(poly,source,targets,rate):
    derivative=PPoly.from_bernstein_basis(poly).derivative();roots=derivative.derivative().roots(extrapolate=False);points=np.r_[poly.x,roots[np.isfinite(roots)]];values=derivative(points);minimum=float(values.min());maximum=float(values.max())
    return dict(source_per_music_rate=[minimum,maximum],allowed_source_per_music_rate=[1/rate[1],1/rate[0]],legal=minimum>=1/rate[1]-1e-9 and maximum<=1/rate[0]+1e-9,anchor_error_frames=float(np.max(abs(poly(targets)-source))),endpoint_d1_error=float(np.max(abs(poly(np.array(targets)[[0,-1]],1)-1))),endpoint_d2_error=float(np.max(abs(poly(np.array(targets)[[0,-1]],2)))))

def smooth_map(source,desired,cfg):
    source=np.asarray(source,float);desired=np.asarray(desired,float)
    if len(source)<2 or source.shape!=desired.shape or not np.isfinite(source).all() or not np.isfinite(desired).all() or np.any(np.diff(source)<=0):raise ValueError('Ordered finite phase support')
    desired=source+np.clip(desired-source,-cfg['max_phase_displacement_frames'],cfg['max_phase_displacement_frames']);desired[[0,-1]]=source[[0,-1]]
    for scale in cfg['shrink_scales']:
        target=np.rint(source+scale*(desired-source));target[[0,-1]]=source[[0,-1]]
        if np.any(np.diff(target)<=0):continue
        poly=BPoly.from_derivatives(target,[[s,1.,0.] for s in source],extrapolate=False);cert=certificate(poly,source,target,cfg['music_per_source_rate'])
        if cert['legal']:return poly,dict(source=source.tolist(),target=target.tolist(),scale=scale,certificate=cert)
    raise ValueError('Identity map must be feasible')

def source_phases(R,a,b,cfg):
    part=R[a:b];v=Rotation.from_matrix((part[:-1].swapaxes(-1,-2)@part[1:]).reshape(-1,3,3)).magnitude().reshape(-1,22)[:,UPPER].mean(-1)*30;speed=np.r_[0.,v]
    if np.max(speed)<cfg['minimum_motion_speed_radps']:return [],speed
    ids,_=find_peaks(speed,distance=19,prominence=max(.05,.15*float(np.std(speed))));phases=[dict(frame=int(a+i),type='upper_speed_peak',semantic_truth='UNKNOWN') for i in ids if 12<i<len(speed)-13];return phases,speed

def maps(R,beats,cfg):
    n=len(R);linear=np.arange(n,dtype=float);smooth=linear.copy();records=[]
    for a,b in cfg['core_intervals']:
        phases,speed=source_phases(R,a,b,cfg);warp=multi_warp([a,b],[a,b],phases,beats)
        if warp is None:raise ValueError('Identity duration violates existing E1-W')
        w=np.array(warp['time_warp']);ticks=np.arange(a,b);linear[a:b]=np.interp(ticks,w[:,1],w[:,0]);poly,record=smooth_map(w[:,0],w[:,1],cfg)
        if phases:smooth[a:b]=poly(ticks)
        record.update(interval=[a,b],phases=phases,original_e1w=warp,static_preserved=not bool(phases),maximum_observed_speed_radps=float(speed.max()));records.append(record)
    smooth[np.abs(smooth-np.rint(smooth))<1e-10]=np.rint(smooth[np.abs(smooth-np.rint(smooth))<1e-10])
    protected=np.ones(n,bool)
    for a,b in cfg['core_intervals']:protected[a:b]=False
    assert np.array_equal(smooth[protected],np.arange(n)[protected]) and np.all(np.diff(smooth)>0)
    return linear,smooth,records
