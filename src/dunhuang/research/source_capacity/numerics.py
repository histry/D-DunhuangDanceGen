"""Derivative checks are diagnostics, not universal physical control ratios."""
import numpy as np

def cosine(a,b):
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    scale=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/scale) if scale>1e-25 else None

def check_derivatives(f,zero,direction,target,steps):
    import torch
    with torch.enable_grad():
        _,jvp=torch.autograd.functional.jvp(f,zero,direction,create_graph=False)
        z=zero.detach().clone().requires_grad_();y=f(z)
        vjp=torch.autograd.grad((y*target).sum(),z)[0]
    left=float((jvp*target).sum());right=float((vjp*direction).sum())
    records=[]
    with torch.no_grad():
        for h in steps:
            fd=(f(zero+h*direction)-f(zero-h*direction))/(2*h)
            error=float((fd-jvp).norm()/jvp.norm().clamp_min(1e-12))
            records.append(dict(h=h,relative_error=error,cosine=cosine(fd.cpu().numpy(),jvp.detach().cpu().numpy())))
    return jvp.detach(),vjp.detach(),dict(jvp_norm=float(jvp.norm()),input_norm=float(direction.norm()),
        rho=float(jvp.norm()/direction.norm().clamp_min(1e-12)),direction_cosine=cosine(jvp.detach().cpu().numpy(),target.detach().cpu().numpy()),
        adjoint_relative_error=abs(left-right)/max(abs(left),abs(right),1e-12),finite_differences=records)

def trusted(record,cfg):
    fd=record["finite_differences"]
    return bool(record["adjoint_relative_error"]<=cfg["max_adjoint_relative_error"] and
        sum(v["relative_error"]<=cfg["max_fd_relative_error"] and v["cosine"] is not None and v["cosine"]>=cfg["min_fd_cosine"] for v in fd)>=2)
