"""Fixed-parameter physical-frame KF/RTS. Never smooth rendered XYZ or claim contact."""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import c2_envelope

def _model(dt,sigma_accel,sigma_obs):
    if not dt>0 or not sigma_accel>0 or not sigma_obs>0:raise ValueError('Positive physical time and noise')
    F=np.array([[1,dt],[0,1.]])
    g=np.array([dt*dt/2,dt]);Q=sigma_accel**2*np.outer(g,g)+np.eye(2)*1e-12
    return F,Q,float(sigma_obs)**2

def linear(z,sigma_obs,sigma_accel,dt=1/30,smooth=True):
    z=np.asarray(z,float)
    if z.ndim<2 or len(z)<3 or not np.isfinite(z).all():raise ValueError('Actual finite time series')
    shape=z.shape;z=z.reshape(len(z),-1);F,Q,V=_model(dt,sigma_accel,sigma_obs)
    m=np.zeros((len(z),2,z.shape[1]));P=np.zeros((len(z),2,2));pred=m.copy();PP=P.copy()
    state=np.stack([z[0],(z[1]-z[0])/dt]);cov=np.diag([V,sigma_accel**2])
    for k in range(len(z)):
        if k:state=F@state;cov=F@cov@F.T+Q
        pred[k]=state;PP[k]=cov;gain=cov[:,0]/(cov[0,0]+V)
        state=state+gain[:,None]*(z[k]-state[0]);H=np.eye(2)-np.outer(gain,[1.,0.])
        cov=H@cov@H.T+V*np.outer(gain,gain);m[k]=state;P[k]=cov
    if smooth:
        for k in range(len(z)-2,-1,-1):
            J=np.linalg.solve(PP[k+1],F@P[k]).T
            m[k]+=J@(m[k+1]-pred[k+1])
    return m[:,0].reshape(shape)

def rotation_filter(R,sigma_obs,sigma_accel,dt=1/30,smooth=True):
    """Right-error relinearized orientation/velocity filter. Diagonal-axis approximation, not exact Lie EKF."""
    R=np.asarray(R,float);N,J=R.shape[:2];F,Q,V=_model(dt,sigma_accel,sigma_obs)
    if R.shape!=(N,J,3,3) or N<3 or not np.isfinite(R).all():raise ValueError('Finite rotation series required')
    filtered=np.empty_like(R);predicted=np.empty_like(R);omega=np.zeros((N,J,3));wp=omega.copy();P=np.empty((N,2,2));PP=P.copy()
    current=R[0].copy();w=Rotation.from_matrix(R[0].swapaxes(-1,-2)@R[1]).as_rotvec()/dt;cov=np.diag([V,sigma_accel**2])
    for k in range(N):
        if k:current=current@Rotation.from_rotvec(w*dt).as_matrix();cov=F@cov@F.T+Q
        predicted[k]=current;wp[k]=w;PP[k]=cov
        error=Rotation.from_matrix(current.swapaxes(-1,-2)@R[k]).as_rotvec();gain=cov[:,0]/(cov[0,0]+V)
        current=current@Rotation.from_rotvec(error*gain[0]).as_matrix();w=w+error*gain[1]
        H=np.eye(2)-np.outer(gain,[1.,0.]);cov=H@cov@H.T+V*np.outer(gain,gain)
        filtered[k]=current;omega[k]=w;P[k]=cov
    if smooth:
        for k in range(N-2,-1,-1):
            gain=np.linalg.solve(PP[k+1],F@P[k]).T
            dr=Rotation.from_matrix(predicted[k+1].swapaxes(-1,-2)@filtered[k+1]).as_rotvec()
            dw=omega[k+1]-wp[k+1]
            filtered[k]=filtered[k]@Rotation.from_rotvec(gain[0,0]*dr+gain[0,1]*dw).as_matrix()
            omega[k]+=gain[1,0]*dr+gain[1,1]*dw
    return filtered

def residual_rts(R,reference,intervals,budget,cfg):
    """Filter small spatial residuals in each continuous source support separately; never cross event cuts."""
    out=np.asarray(R,float).copy();limit=np.asarray(budget,float)
    for a,b in intervals:
        a,b=int(a),int(b)
        if b-a<3:continue
        # Spatial residual coordinates live in fixed parent-joint axes; convert back to right residual for the budget.
        d=R[a:b]@reference[a:b].swapaxes(-1,-2);v=Rotation.from_matrix(d.reshape(-1,3,3)).as_rotvec().reshape(b-a,22,3)
        if np.linalg.norm(v,axis=-1).max()>=np.pi-.1:raise ValueError('Residual chart branch is ambiguous')
        filtered=linear(v,cfg['kalman_angle_sigma_rad'],cfg['kalman_angular_accel_sigma'])
        proposal=Rotation.from_rotvec(filtered.reshape(-1,3)).as_matrix().reshape(-1,22,3,3)@reference[a:b]
        rv=Rotation.from_matrix((R[a:b].swapaxes(-1,-2)@proposal).reshape(-1,3,3)).as_rotvec().reshape(b-a,22,3)
        envelope=c2_envelope(b-a,[0,b-a],min(12,max(1,(b-a-1)//2)))
        candidate=R[a:b]@Rotation.from_rotvec((rv*envelope[:,None,None]).reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
        from dunhuang.research.complete_r36.core import bounded_rotations
        out[a:b]=bounded_rotations(reference[a:b],candidate,limit[a:b,None])
    return out
