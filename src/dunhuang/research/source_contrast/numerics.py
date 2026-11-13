"""Shared finite-difference forwards, separate physical and source qualification."""
import torch
from dunhuang.research.source_capacity.numerics import cosine,trusted
def check(f,zero,direction,cfg):
    if float(direction.double().norm())==0:
        return None,dict(status="NOT_EVALUABLE_ZERO_COMMON_DOSE",numeric_trusted=False,physical=None,source=None)
    _,jp=torch.autograd.functional.jvp(f,zero,direction)
    records={}
    for name,part in (("physical",slice(0,8250)),("source",slice(8250,None))):
        target=torch.zeros_like(jp);target[part]=jp[part].detach()/jp[part].norm().clamp_min(1e-12)
        z=zero.detach().clone().requires_grad_();value=f(z)
        g=torch.autograd.grad((value*target).sum(),z)[0]
        left=float((jp*target).sum());right=float((g*direction).sum())
        records[name]=dict(jvp_norm=float(jp[part].norm()),adjoint_relative_error=abs(left-right)/max(abs(left),abs(right),1e-12),finite_differences=[])
    with torch.no_grad():
        for h in cfg["fd_steps"]:
            fd=(f(zero+h*direction)-f(zero-h*direction))/(2*h)
            for name,part in (("physical",slice(0,8250)),("source",slice(8250,None))):
                records[name]["finite_differences"].append(dict(h=h,relative_error=float((fd[part]-jp[part]).norm()/jp[part].norm().clamp_min(1e-12)),
                    cosine=cosine(fd[part].cpu().numpy(),jp[part].detach().cpu().numpy())))
    return jp.detach(),dict(status="MEASURED_NONZERO",numeric_trusted=all(trusted(v,cfg) for v in records.values()),**records)
