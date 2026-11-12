"""Shared target coordinates and weight identity, independent of experiment trainers."""
import hashlib
import numpy as np
from dunhuang.contracts import MotionAdapter
from .probe import upperbody_probe_target
from .soft_reference import resample_native

def phase_target(native,scale,bias,knot=75,partial=False):
    """The same geodesic source-time operator as inference, never linear Rot6D."""
    positions=np.interp(np.arange(150),[0,75,150],[0,knot,150])
    sampled=resample_native(native,np.minimum(positions,149))
    origin=native[0,4:7].copy();origin[1]=0.
    normalized=MotionAdapter.to_edge(sampled,origin)*scale+bias
    valid=np.isfinite(normalized)&(abs(normalized)<=1+32*np.finfo(np.float32).eps)
    target=upperbody_probe_target(normalized,valid) if partial else (normalized if valid.all() else None)
    return target, (positions/150).astype(np.float32)[:,None]

def digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
