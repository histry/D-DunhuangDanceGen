"""Small coefficient-space convex projection; no dense condition Jacobian."""
import numpy as np
from scipy.optimize import minimize,nnls

ROOT=450
CONTACT=600
OTHER=2400

def constraints(a,gram,z,margin):
    root=a[:ROOT];contact=a[ROOT:ROOT+CONTACT]*np.sqrt(CONTACT)
    other=a[ROOT+CONTACT:].reshape(OTHER,3,-1)*np.sqrt(OTHER)
    r=root@z;c=contact@z;o=np.einsum("ijk,k->ij",other,z)
    norms=np.linalg.norm(o,axis=1);unit=np.divide(o,norms[:,None],out=np.zeros_like(o),where=norms[:,None]>1e-15)
    value=np.r_[margin**2-r@r,margin-c,margin+c,margin-norms.mean(),1-z@gram@z]
    jac=np.concatenate([(-2*r@root)[None],-contact,contact,(-np.einsum("ij,ijk->k",unit,other)/OTHER)[None],(-2*z@gram)[None]])
    return value,jac

def project(a,gram,target,cfg):
    a=np.asarray(a,float);gram=np.asarray(gram,float);target=np.asarray(target,float)
    k=len(target)
    if a.shape!=(ROOT+CONTACT+3*OTHER,k) or gram.shape!=(k,k) or not np.isfinite(a).all() or not np.isfinite(gram).all() or not np.isfinite(target).all():raise ValueError("Finite small-basis physical derivatives required")
    if np.linalg.eigvalsh(gram).min()<=1e-9:raise ValueError("Independent direction basis required")
    if not 0<cfg["linear_safety_fraction"]<=1:raise ValueError("Safety fraction cannot relax physical bounds")
    margin=cfg["linear_safety_fraction"]
    objective=lambda z:.5*z@gram@z-target@z
    gradient=lambda z:gram@z-target
    result=minimize(objective,np.zeros(k),jac=gradient,method="SLSQP",
        constraints=dict(type="ineq",fun=lambda z:constraints(a,gram,z,margin)[0],jac=lambda z:constraints(a,gram,z,margin)[1]),
        options=dict(maxiter=cfg["max_solver_iterations"],ftol=cfg["solver_ftol"],disp=False))
    value,jac=constraints(a,gram,result.x,margin);violation=float(max(0,-value.min()))
    accepted=bool(result.success and np.isfinite(result.x).all() and violation<=cfg["linear_tolerance"])
    active=value<1e-5
    try:
        if active.any():lam,_=nnls(jac[active].T,gradient(result.x),maxiter=1000);stationarity=float(np.linalg.norm(gradient(result.x)-jac[active].T@lam))
        else:stationarity=float(np.linalg.norm(gradient(result.x)))
    except RuntimeError:stationarity=None
    z=result.x if accepted else np.zeros(k)
    return z,dict(status="SOLVED_FINITE_SUBSPACE" if accepted else "UNKNOWN_SOLVER_OR_LINEAR_FEASIBILITY",
        scipy_status=int(result.status),iterations=int(result.nit),max_linear_violation=violation,
        stationarity_residual=stationarity,coefficient_norm=float(np.sqrt(max(0,z@gram@z))),
        rank=k,global_infeasibility_claimed=False,safety_fraction=margin,solver_message=str(result.message))

def basis(f,zero,seed,support,radius,rank):
    import torch
    from dunhuang.research.source_capacity.numerics import support_value
    directions=[];images=[];v=seed.detach()
    for _ in range(rank):
        v=support_value(v,support)
        for repeat in range(2):
            for old in directions:v=v-float((v.double()*old.double()).sum())*old
        norm=float(v.double().norm())
        if norm<1e-10:break
        v=v/norm;directions.append(v.detach())
        _,image=torch.autograd.functional.jvp(f,zero,radius*v)
        images.append(image.detach())
        with torch.enable_grad():
            z=zero.detach().clone().requires_grad_();y=f(z)
            v=torch.autograd.grad((y*image.detach()).sum(),z)[0].detach()
    if not directions:raise ValueError("No resolvable source direction")
    flat=torch.stack([v.flatten().double() for v in directions])
    gram=(flat@flat.T).cpu().numpy()
    a=torch.stack(images,-1).cpu().numpy().astype(float)
    target=(flat@seed.flatten().double()/radius).cpu().numpy()
    return directions,a,gram,target

def physical_coordinates(rt,pred,base_rot):
    import torch
    from dunhuang.generation.event_geometry import _matrix
    raw=(pred-torch.as_tensor(rt.bias,device=pred.device))/torch.as_tensor(rt.scale,device=pred.device)
    rotation=_matrix(raw[...,7:].reshape(1,150,24,6)).transpose(-1,-2)[0]
    other=[j for j in range(24) if not 12<=j<20]
    relative=base_rot[:,other].transpose(-1,-2)@rotation[:,other]
    tangent=torch.stack((relative[...,2,1]-relative[...,1,2],relative[...,0,2]-relative[...,2,0],relative[...,1,0]-relative[...,0,1]),-1)/2
    return torch.cat((raw[0,:,4:7].flatten()/(.01*np.sqrt(ROOT)),raw[0,:,:4].flatten()/(.05*np.sqrt(CONTACT)),tangent.flatten()/(.01*np.sqrt(OTHER))))
