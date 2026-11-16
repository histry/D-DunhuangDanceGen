"""Stable box-constrained equivalent row6D; exact convex 2D projections, no motion smoothing."""
import itertools
import numpy as np

def scaler_box(scale,offset,training_min,training_max,margin):
    scale=np.asarray(scale);offset=np.asarray(offset)
    if scale.shape!=(139,) or offset.shape!=(139,) or (scale<=0).any():raise ValueError("139D positive affine scaler required")
    constant=np.asarray(training_min)[7:]==np.asarray(training_max)[7:]
    interior=np.where(constant,0.,margin)
    return (-1+interior-offset[7:])/scale[7:],(1-interior-offset[7:])/scale[7:],constant

def _project(q,A,c,vertices,vertex_valid):
    norm=(A*A).sum(-1)
    violations=np.einsum("nkd,nd->nk",A,q)-c
    lines=q[:,None]-violations[:,:,None]*A/np.maximum(norm[:,:,None],1e-24)
    candidates=np.concatenate((q[:,None],vertices,lines),1)
    valid=(np.einsum("nkd,nvd->nvk",A,candidates)<=c[:,None]+1e-9).all(-1)
    valid[:,1:1+vertices.shape[1]] &= vertex_valid
    distances=((candidates-q[:,None])**2).sum(-1)
    distances=np.where(valid,distances,np.inf)
    feasible=np.isfinite(distances).any(-1)
    choice=np.argmin(distances,axis=1)
    value=candidates[np.arange(len(q)),choice]
    return np.where(feasible[:,None],value,0.),feasible

def solve(rotations,lower,upper,minimum=.05,iterations=64,smooth=.25,curvature=.025):
    R=np.asarray(rotations,dtype=np.float64)
    if R.ndim!=4 or R.shape[1:]!=(22,3,3):raise ValueError("T22 SO3 matrices required")
    if not np.isfinite(R).all() or np.max(abs(R@R.swapaxes(-1,-2)-np.eye(3)))>1e-5:raise ValueError("Nonrotation input")
    l=np.asarray(lower,dtype=np.float64).reshape(22,6);u=np.asarray(upper,dtype=np.float64).reshape(22,6)
    if not np.isfinite(l).all() or not np.isfinite(u).all() or (l>=u).any() or minimum<=0:raise ValueError("Finite positive-width box and stability floor required")
    T=len(R);r1=R[:,:,0];r2=R[:,:,1];zero=abs(r1)<1e-12
    safe=np.where(zero,1,r1)
    lo=np.minimum(l[None,:,:3]/safe,u[None,:,:3]/safe)
    hi=np.maximum(l[None,:,:3]/safe,u[None,:,:3]/safe)
    lo=np.where(zero,-np.inf,lo);hi=np.where(zero,np.inf,hi)
    alo=np.maximum(lo.max(-1),minimum);ahi=hi.min(-1)
    avalid=(alo<=ahi)&(~(zero&((l[None,:,:3]>0)|(u[None,:,:3]<0))).any(-1))
    basis=np.stack((r1,r2),-1)
    A=np.concatenate((basis,-basis,np.broadcast_to(np.array([0.,-1.]),(T,22,1,2))),2).reshape(-1,7,2)
    c=np.concatenate((np.broadcast_to(u[None,:,3:],(T,22,3)),-np.broadcast_to(l[None,:,3:],(T,22,3)),np.full((T,22,1),-minimum)),2).reshape(-1,7)
    vertices=[];valid=[]
    for i,j in itertools.combinations(range(7),2):
        a,b=A[:,i],A[:,j];det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        safe_det=np.where(abs(det)>1e-12,det,1.)
        v=np.stack(((c[:,i]*b[:,1]-a[:,1]*c[:,j])/safe_det,(a[:,0]*c[:,j]-c[:,i]*b[:,0])/safe_det),-1)
        good=(abs(det)>1e-12)&(np.einsum("nkd,nd->nk",A,v)<=c+1e-9).all(-1)
        vertices.append(v);valid.append(good)
    vertices=np.stack(vertices,1);valid=np.stack(valid,1)
    q,feasible=_project(np.broadcast_to(np.array([0.,1.]),(T*22,2)),A,c,vertices,valid)
    feasible=feasible.reshape(T,22)&avalid
    coeff=np.concatenate((np.clip(np.ones((T,22,1)),alo[:,:,None],np.maximum(ahi,alo)[:,:,None]),q.reshape(T,22,2)),2)
    coeff=np.where(feasible[:,:,None],coeff,0.)
    initial=coeff.copy();canonical=np.broadcast_to(np.array([1.,0.,1.]),coeff.shape)
    step=1/(1+4*smooth+16*curvature)
    for _ in range(iterations):
        grad=(coeff-canonical)*feasible[:,:,None]
        d=np.diff(coeff,axis=0)*(feasible[1:]&feasible[:-1])[:,:,None]
        grad[:-1]-=smooth*d;grad[1:]+=smooth*d
        d=np.diff(coeff,n=2,axis=0)*(feasible[2:]&feasible[1:-1]&feasible[:-2])[:,:,None]
        grad[:-2]+=curvature*d;grad[1:-1]-=2*curvature*d;grad[2:]+=curvature*d
        v=coeff-step*grad
        alpha=np.clip(v[:,:,0],alo,np.maximum(ahi,alo))
        b,_=_project(v[:,:,1:].reshape(-1,2),A,c,vertices,valid)
        coeff=np.concatenate((alpha[:,:,None],b.reshape(T,22,2)),2)
        coeff=np.where(feasible[:,:,None],coeff,0.)
    code=np.concatenate((coeff[:,:,:1]*r1,coeff[:,:,1:2]*r1+coeff[:,:,2:3]*r2),-1)
    code=np.where(feasible[:,:,None],code,np.nan).astype(np.float32)
    return dict(code=code,feasible=feasible,coefficients=np.where(feasible[:,:,None],coeff,np.nan),
        initial_coefficients=np.where(feasible[:,:,None],initial,np.nan),alpha_lower=alo,alpha_upper=ahi,
        alpha_feasible=avalid,beta_gamma_feasible=valid.any(1).reshape(T,22),minimum=minimum)
