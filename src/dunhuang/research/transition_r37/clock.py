"""Explicit versioned frame-centre reader shared by content, preview and generation."""
import numpy as np
from dunhuang.planning.time_map import source_positions as legacy_positions

FRAME_CENTRES='ACTUAL_FRAME_CENTERS_WITH_HALF_OPEN_SUPPORT'

def source_positions(event, ticks):
    version=event.get('time_contract') or event.get('clock_contract')
    if version is None:
        return legacy_positions(event,ticks)
    if version!=FRAME_CENTRES:raise ValueError('Unknown time contract')
    lo,hi=event['source_interval'];a,b=event['music_interval']
    w=np.asarray(event['time_warp'],float);t=np.asarray(ticks,float)
    if not 0<=lo<hi-1 or not 0<=a<b-1 or w.ndim!=2 or w.shape[1]!=2 or len(w)<2:raise ValueError('Nonempty frame-centre support')
    if not np.isfinite(w).all() or (np.diff(w,axis=0)<=0).any():raise ValueError('Strictly monotone finite warp')
    if not np.array_equal(w[0],[lo,a]) or not np.array_equal(w[-1],[hi-1,b-1]):raise ValueError('Frame-centre endpoints differ')
    rates=np.diff(w[:,1])/np.diff(w[:,0])
    if ((rates<.75-1e-6)|(rates>1.33+1e-6)).any():raise ValueError('Rate outside registered limits')
    if t.ndim!=1 or not np.isfinite(t).all() or ((t<a)|(t>=b)).any():raise ValueError('Requested time outside support')
    return np.interp(t,w[:,1],w[:,0])
