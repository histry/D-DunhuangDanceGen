"""SO(3) chart quintic Hermite baseline and explicit local necessary bounds."""
import numpy as np
from scipy.spatial.transform import Rotation

def quintic_bridge(rotations,gap):
    R=np.asarray(rotations,float);a,b=map(int,gap)
    if not 2<=a<b-2<=len(R)-3:raise ValueError('Two actual derivative neighbors required')
    anchor=R[a]
    points=np.array([a-2,a-1,a,b-1,b,b+1]);relative=anchor[None].swapaxes(-1,-2)@R[points]
    v=Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(6,22,3)
    if np.linalg.norm(v,axis=-1).max()>np.pi-.05:raise ValueError('Ambiguous SO3 chart; cannot manufacture transition')
    T=b-1-a
    constraints=np.stack([v[2],(v[2]-v[0])/2*T,(v[2]-2*v[1]+v[0])*T*T,
                          v[3],(v[5]-v[3])/2*T,(v[5]-2*v[4]+v[3])*T*T])
    A=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,2,0,0,0],
                [1,1,1,1,1,1],[0,1,2,3,4,5],[0,0,2,6,12,20]],float)
    coefficients=np.linalg.solve(A,constraints.reshape(6,-1))
    u=np.linspace(0,1,b-a);poly=np.stack([u**i for i in range(6)],1)@coefficients
    proposal=anchor[None]@Rotation.from_rotvec(poly.reshape(-1,3)).as_matrix().reshape(b-a,22,3,3)
    out=R.copy();out[a:b]=proposal
    return out

def c2_envelope(n,gap,ramp=12):
    a,b=map(int,gap)
    if not 0<=a<b<=n or ramp<1:raise ValueError('Editable range')
    d=np.minimum(np.arange(n)-a,b-1-np.arange(n));u=np.clip(d/ramp,0,1)
    return u**3*(10-15*u+6*u*u)

def necessary_jump(reference,budget,boundary):
    d=Rotation.from_matrix(reference[boundary-1].swapaxes(-1,-2)@reference[boundary]).magnitude()
    return dict(reference_angle_rad=d.tolist(),necessary_remaining_angle_rad=np.maximum(0,d-budget[boundary-1]-budget[boundary]).tolist(),
                scope='ONE_STEP_LOCAL_SO3_NECESSARY_BOUND_NOT_GLOBAL_FEASIBILITY')
