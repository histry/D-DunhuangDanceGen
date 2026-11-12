"""Explicit shared q-distribution support and source-conditioned preference."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices
from dunhuang.research.necessity.factors import ddim_times

def angles(a,b,mask):
    a=np.asarray(a);b=np.asarray(b);mask=np.asarray(mask)
    if a.shape!=(150,151) or b.shape!=a.shape or mask.shape!=(150,8) or mask.dtype!=bool or not mask.any():raise ValueError("Actual common upper8 support required")
    ra=matrices(a[:,79:127].reshape(150,8,6));rb=matrices(b[:,79:127].reshape(150,8,6))
    return Rotation.from_matrix((ra.swapaxes(-1,-2)@rb)[mask]).magnitude()

def response(a,b,ra,rb,mask):
    target=float(angles(ra,rb,mask).mean());actual=float(angles(a,b,mask).mean())
    return dict(output_separation_rad=actual,reference_separation_rad=target,
        normalized_response=actual/(target+1e-12),reference_observed=True)

def noise_support(target_a,target_b,alpha,cfg):
    alpha=np.asarray(alpha,dtype=float)
    if alpha.shape!=(1000,) or not np.isfinite(alpha).all() or not ((alpha>0)&(alpha<1)).all():raise ValueError("Actual finite nonzero DDPM signal required")
    norm=float(np.square(np.asarray(target_a,dtype=float)-target_b).sum())
    times=np.sort(ddim_times());snr=alpha/(1-alpha);kl=.5*snr*norm
    keep=times[(times>=cfg["minimum_ddim_t"])&(snr[times]<=cfg["max_snr"])&(kl[times]<=cfg["max_mean_kl"])]
    return keep,dict(squared_mean_distance=norm,max_selected_kl=float(kl[keep].max()) if len(keep) else None,
        max_selected_snr=float(snr[keep].max()) if len(keep) else None)

def cf_loss(predicted,reference,mask,scale,bias,cfg):
    import torch
    from dunhuang.generation.event_geometry import _matrix
    from dunhuang.generation.source_following import geodesic_squared
    if predicted.shape!=(2,150,151) or reference.shape!=predicted.shape or mask.shape!=(150,8) or mask.dtype!=torch.bool or not mask.any():raise ValueError("One qualified pair with common observed support required")
    pr=_matrix(((predicted-bias)/scale)[...,79:127].reshape(2,150,8,6)).transpose(-1,-2)
    identity=torch.tensor([1,0,0,0,1,0],device=reference.device,dtype=reference.dtype)
    safe=torch.where(mask[None,...,None],reference[...,79:127].reshape(2,150,8,6),identity)
    rr=_matrix(safe)
    d=torch.stack([torch.stack([geodesic_squared(pr[i],rr[j])[mask].mean() for j in range(2)]) for i in range(2)])
    margin=(cfg["cf_margin_fraction"]*geodesic_squared(rr[0],rr[1])[mask].mean()).clamp(max=cfg["cf_margin_max_squared_rad"]).detach()
    value=((margin+d[0,0]-d[0,1]).relu()+(margin+d[1,1]-d[1,0]).relu())/2
    return value,margin

def role_noise_admitted(role,support,minimum):
    if role not in ("train","val"):raise ValueError("Only explicit Train/Val roles admitted")
    return role=="val" or len(support)>=minimum
