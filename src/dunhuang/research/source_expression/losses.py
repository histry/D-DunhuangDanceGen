"""Inference-available pose, adjacent SO(3) dynamics and source-stage objectives."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices

def scales(available,cfg):
    poses=[];velocities=[]
    for safe,known,stage,weights in available:
        r=matrices(safe).astype(float)
        poses.extend(Rotation.from_matrix(r[known]).magnitude()**2)
        valid=known[1:]&known[:-1]
        velocities.extend(Rotation.from_matrix((r[:-1].swapaxes(-1,-2)@r[1:])[valid]).magnitude()**2*900)
    if not len(poses) or not len(velocities):raise ValueError("Observed Train pose/dynamic calibration required")
    return dict(pose=max(float(np.mean(poses)),cfg["pose_floor_squared_rad"]),velocity=max(float(np.mean(velocities)),cfg["velocity_floor_squared_radps"]),
        pose_samples=len(poses),adjacent_samples=len(velocities),scope="old four Train pairs, reference inputs only",model_fit_calls=0)

def reference_data(available,seed,shuffle=False,device='cuda'):
    import torch
    from dunhuang.generation.event_geometry import _matrix
    safe,known,stage,weights=available
    r=_matrix(torch.from_numpy(safe).to(device))
    increment=r[:-1].transpose(-1,-2)@r[1:]
    valid=known[1:]&known[:-1];stage=stage&known
    if shuffle:
        # Permute only visible targets within each joint; no invented source trajectory.
        rng=np.random.default_rng(seed);increment=increment.clone();stage=stage.copy()
        for j in range(8):
            ids=np.flatnonzero(valid[:,j]);donors=rng.permutation(ids)
            increment[torch.as_tensor(ids,device=device),j]=increment[torch.as_tensor(donors,device=device),j].clone()
            frames=np.flatnonzero(known[:,j]);stage[frames,j]=stage[rng.permutation(frames),j]
    return r,increment,torch.from_numpy(known).to(device),torch.from_numpy(valid).to(device),torch.from_numpy(stage).to(device)

def components(rt,pred,data):
    import torch
    from dunhuang.generation.event_geometry import _matrix
    from dunhuang.generation.source_following import geodesic_squared
    r,inc,known,valid,stage=data
    raw=(pred-torch.as_tensor(rt.bias,device=pred.device))/torch.as_tensor(rt.scale,device=pred.device)
    pr=_matrix(raw[...,79:127].reshape(1,150,8,6)).transpose(-1,-2)[0]
    pose=geodesic_squared(pr,r)
    pi=pr[:-1].transpose(-1,-2)@pr[1:]
    velocity=geodesic_squared(pi,inc)
    return dict(pose=(pose*known).sum()/known.sum(),velocity=velocity[valid].mean()*900,
        event=pose[stage].mean() if stage.any() else pose[known].mean()*0)

def expression_loss(rt,pred,data,norm,cfg,arm):
    c=components(rt,pred,data)
    if arm=="D0":return c["pose"]
    return c["pose"]/norm["pose"]+cfg["velocity_weight"]*c["velocity"]/norm["velocity"]+(cfg["event_weight"]*c["event"]/norm["pose"] if arm in ("D2","D3") else 0)

def equalize(directions):
    # A non-expansive common dose makes all arms exactly comparable in FP32.
    import torch
    norm=min(float(v.double().norm()) for v in directions.values())
    return {k:v*(norm/float(v.double().norm())) if float(v.double().norm())>0 else torch.zeros_like(v) for k,v in directions.items()},norm
