"""Constant-acceleration RTS with exact endpoint states/positions inside the solver.

Independent coordinates in a fixed parent-local spatial SO3 residual chart; not
an invariant Lie-group RTS. Covariances are regularizers, not calibrated confidence.
"""
import numpy as np
from scipy.spatial.transform import Rotation

def model(dt,sigma_jerk):
    if dt<=0 or sigma_jerk<=0:raise ValueError('Positive physical frame time and process scale')
    F=np.array([[1,dt,dt*dt/2],[0,1,dt],[0,0,1.]])
    Q=sigma_jerk**2*np.array([[dt**5/20,dt**4/8,dt**3/6],[dt**4/8,dt**3/3,dt**2/2],[dt**3/6,dt**2/2,dt]])
    return F,Q

def constrained_rts(values,sigma_observation,sigma_jerk,dt=1/30,boundary=3):
    y=np.asarray(values,float);shape=y.shape;n=len(y)
    if y.ndim<2 or n<2*boundary+1 or boundary<1 or sigma_observation<=0 or not np.isfinite(y).all():raise ValueError('Finite residual and explicit constraint support')
    y=y.reshape(n,-1);F,Q=model(dt,sigma_jerk);V=float(sigma_observation)**2
    filtered=np.zeros((n,3,y.shape[1]));predicted=filtered.copy();P=np.zeros((n,3,3));PP=P.copy();state=np.zeros((3,y.shape[1]));cov=np.zeros((3,3))
    for t in range(n):
        if t:state=F@state;cov=F@cov@F.T+Q
        predicted[t]=state;PP[t]=cov
        if t in (0,n-1):state=np.zeros_like(state);cov=np.zeros_like(cov)
        else:
            hard=t<boundary or t>=n-boundary;noise=0. if hard else V;obs=np.zeros(y.shape[1]) if hard else y[t]
            gain=cov[:,0]/(cov[0,0]+noise);state=state+gain[:,None]*(obs-state[0]);A=np.eye(3)-np.outer(gain,[1.,0,0]);cov=A@cov@A.T+noise*np.outer(gain,gain);cov=(cov+cov.T)/2
        filtered[t]=state;P[t]=cov
    for t in range(n-2,-1,-1):
        gain=np.linalg.solve(PP[t+1],F@P[t]).T;filtered[t]+=gain@(filtered[t+1]-predicted[t+1])
    if max(abs(filtered[0]).max(),abs(filtered[-1]).max(),abs(filtered[:boundary,0]).max(),abs(filtered[-boundary:,0]).max())>1e-8:raise ValueError('Endpoint equality numerical failure')
    # Only remove floating roundoff in already constrained quantities.
    filtered[0]=0;filtered[-1]=0;filtered[:boundary,0]=0;filtered[-boundary:,0]=0
    return filtered[:,0].reshape(shape),dict(endpoint_state_max_abs=float(max(abs(filtered[0]).max(),abs(filtered[-1]).max())),boundary_position_samples=boundary,method='EXACT_EQUALITY_OBSERVATIONS_AND_FIXED_ENDPOINT_STATES_THEN_RTS')

def global_bound(values,budget):
    v=np.asarray(values,float);bound=np.broadcast_to(np.asarray(budget,float),v.shape[:-1]);norm=np.linalg.norm(v,axis=-1);ok=norm>1e-12
    ratio=float(min(1.,np.min(bound[ok]/norm[ok]))) if ok.any() else 1.
    if ratio<0:raise ValueError('Negative residual budget')
    return v*ratio,ratio

def residuals(raw,anchor,gap,calibration,root_scale,root_offset,body_budget):
    from dunhuang.research.transition_r37.evaluate import decode
    from dunhuang.research.transition_r37.geometry import c2_envelope
    x=np.asarray(raw,float);base=np.asarray(anchor,float);a,b=gap;R,root=decode(x);B,base_root=decode(base);out=x.copy();w=c2_envelope(len(x),gap)[a:b]
    root_y=root[a:b]-base_root[a:b];spatial=R[a:b]@B[a:b].swapaxes(-1,-2);angle=Rotation.from_matrix(spatial.reshape(-1,3,3)).as_rotvec().reshape(b-a,22,3)
    if np.linalg.norm(angle,axis=-1).max()>=np.pi-.1:raise ValueError('SO3 residual chart ambiguity')
    rp,rmeta=constrained_rts(root_y,calibration['root_observation_sigma_m'],calibration['root_process_jerk_m_s3'],boundary=3)
    rv,bmeta=constrained_rts(angle,calibration['body_observation_sigma_rad'],calibration['body_process_jerk_rad_s3'],boundary=3)
    rp,root_gain=global_bound(rp,w*calibration['root_budget_m']);rv,body_gain=global_bound(rv,w[:,None]*body_budget)
    # A single scaling preserves every equality and smooth residual shape, unlike framewise clipping.
    lower=(-1-np.asarray(root_offset))/np.asarray(root_scale);upper=(1-np.asarray(root_offset))/np.asarray(root_scale);gamma=1.
    for d in range(3):
        pos=rp[:,d]>1e-12;neg=rp[:,d]<-1e-12
        if pos.any():gamma=min(gamma,float(np.min((upper[d]-base_root[a:b,d][pos])/rp[pos,d])))
        if neg.any():gamma=min(gamma,float(np.min((lower[d]-base_root[a:b,d][neg])/rp[neg,d])))
    gamma=max(0.,min(1.,gamma));rp*=gamma
    smooth_R=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(b-a,22,3,3)@B[a:b]
    out[a:b,4:7]=base_root[a:b]+rp;out[a:b,7:]=smooth_R[...,:2,:].reshape(b-a,132)
    # Only remove roundoff after verifying the solver already meets these equalities.
    if max(abs(out[a:a+3,4:]-base[a:a+3,4:]).max(),abs(out[b-3:b,4:]-base[b-3:b,4:]).max())>1e-6:raise ValueError('Reconstruction violated solved endpoint equalities')
    out[a:a+3,4:]=base[a:a+3,4:];out[b-3:b,4:]=base[b-3:b,4:]
    if not np.array_equal(out[:a],x[:a]) or not np.array_equal(out[b:],x[b:]) or not np.array_equal(out[:,:4],x[:,:4]):raise ValueError('Read-only region/C4 changed')
    return out.astype(raw.dtype),dict(root_constraints=rmeta,body_constraints=bmeta,root_global_budget_gain=root_gain,root_box_gain=gamma,body_global_budget_gain=body_gain,not_confidence_calibrated=True)
