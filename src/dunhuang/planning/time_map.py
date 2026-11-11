"""One physical source-to-music clock for ranking and generation."""
import numpy as np
SCHEMA='event_time_warp_motion74_v1'

def source_positions(event,music_frames):
    lo,hi=event['source_interval'];a,b=event['music_interval']
    warp=np.asarray(event['time_warp'],np.float64);times=np.asarray(music_frames,np.float64)
    if not 0<=lo<hi or not 0<=a<b or warp.ndim!=2 or warp.shape[1]!=2 or len(warp)<2:
        raise ValueError('Invalid mapped event interval')
    if not np.isfinite(warp).all() or np.any(np.diff(warp,axis=0)<=0):
        raise ValueError('Time map must be finite and strictly monotone')
    if not np.array_equal(warp[0],[lo,a]) or not np.array_equal(warp[-1],[hi,b]):
        raise ValueError('Event prompt time map endpoints differ')
    rates=np.diff(warp[:,1])/np.diff(warp[:,0])
    if np.any(rates<.75-1e-6) or np.any(rates>1.33+1e-6):
        raise ValueError('Time-map rate outside registered limits')
    if times.ndim!=1 or not np.isfinite(times).all() or np.any(times<a) or np.any(times>=b):
        raise ValueError('Requested music time outside observed event')
    # A half-open event has no observed pose at hi. Keep its final observation.
    return np.minimum(np.interp(times,warp[:,1],warp[:,0]),hi-1)

def motion_features74(native):
    """Recompute 30Hz physical derivatives AFTER the exact source time warp.

    Same first22 log-rotation layout as paired_alignment_observations_v3.
    Root/contact aggregates stay unobserved; no approximate chain-rule scaling.
    """
    from scipy.spatial.transform import Rotation
    from dunhuang.contracts import matrices,motion151
    x=motion151(native)
    if len(x)<2:raise ValueError('Two real mapped frames required')
    r=matrices(x[:,7:139].reshape(-1,22,6))
    v=np.empty((len(x),22,3),np.float32)
    v[1:]=Rotation.from_matrix((r[:-1].swapaxes(-1,-2)@r[1:]).reshape(-1,3,3)).as_rotvec().reshape(-1,22,3)*30
    v[0]=v[1]
    speed=np.linalg.norm(v,axis=-1);squared=speed**2
    aggregate=np.zeros((len(x),8),np.float32)
    aggregate[:,0]=speed.mean(-1);aggregate[:,1]=np.percentile(speed,90,axis=-1)
    aggregate[:,4]=squared.mean(-1)
    aggregate[:,6]=squared[:,(3,6,9,12,13,14,15,16,17,18,19,20,21)].mean(-1)
    aggregate[:,7]=squared[:,(0,1,2,4,5,7,8,10,11)].mean(-1)
    values=np.concatenate((v.reshape(len(x),66),aggregate),-1)
    mask=np.ones_like(values,dtype=bool);mask[:,(68,69,71)]=False
    return values,mask
