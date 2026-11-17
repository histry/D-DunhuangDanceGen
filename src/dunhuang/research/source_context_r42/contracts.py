"""CPU contracts for genuinely hidden continuous gaps; no contact pseudo-truth."""
import numpy as np
from dunhuang.research.bridge_r38.geometry import observed_bridge,polynomial


def observed_anchor(raw,base,gap,root_observed):
    from dunhuang.research.transition_r37.evaluate import decode
    raw=np.asarray(raw);base=np.asarray(base);R,_=decode(raw);hidden=R.copy();hidden[slice(*gap)]=np.nan;G=observed_bridge(hidden,gap);anchor=base.copy();anchor[:,7:]=G[...,:2,:].reshape(256,132)
    if root_observed:
        root=raw[:,4:7].copy();root[slice(*gap)]=np.nan;anchor[:,4:7]=polynomial(root,gap)
    return anchor


def window_starts(length,train):
    if length<256:raise ValueError('Real continuous256 source support')
    return sorted(set(map(int,np.linspace(0,length-256,12 if train else 3).round())))


def role_check(rows):
    roles={}
    for r in rows:
        uid=r['recording_uid'];roles.setdefault(uid,set()).add(r['role'])
    if any(len(v)>1 for v in roles.values()):raise ValueError('Recording role overlap')
    return dict(recordings=len(roles),train_recordings=sum(v=={'train'} for v in roles.values()),held_recordings=sum(v=={'held'} for v in roles.values()))


def exact_arrays(first,second):
    with np.load(first) as a,np.load(second) as b:
        if set(a.files)!=set(b.files):raise ValueError('Replay keys differ')
        for key in a.files:
            x,y=a[key],b[key]
            if x.dtype!=y.dtype or x.shape!=y.shape:raise ValueError('Replay dtype/shape differs: '+key)
            equal=np.array_equal(x,y,equal_nan=True) if x.dtype.kind in 'fc' else np.array_equal(x,y)
            if not equal:raise ValueError('Replay payload differs: '+key)
