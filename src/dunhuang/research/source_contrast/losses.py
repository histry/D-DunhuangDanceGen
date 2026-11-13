"""Paired identity comparison uses only two provided observed references."""
import torch
from dunhuang.generation.event_geometry import _matrix
from dunhuang.generation.source_following import geodesic_squared
from dunhuang.research.source_expression.losses import components
def errors(rt,pred,own,other):
    mask=own[2]&other[2]
    if not bool(mask.any()):raise ValueError("Observed common support required")
    raw=(pred-torch.as_tensor(rt.bias,device=pred.device))/torch.as_tensor(rt.scale,device=pred.device)
    rotation=_matrix(raw[...,79:127].reshape(1,150,8,6)).transpose(-1,-2)[0]
    # radians, not squared radians: comparable to the registered source readout.
    a=geodesic_squared(rotation,own[0]).sqrt()[mask].mean()
    b=geodesic_squared(rotation,other[0]).sqrt()[mask].mean()
    return torch.stack([a,a-b])
def objective(rt,pred,own,other,dynamic,norm,config,arm):
    identity=errors(rt,pred,own,other)[1]/(norm["pose"]**.5)
    return identity+(config["dynamic_weight"]*components(rt,pred,dynamic)["velocity"]/norm["velocity"] if arm in ("A2","A3") else 0)
