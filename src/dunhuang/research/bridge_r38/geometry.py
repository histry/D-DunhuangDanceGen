"""Observed-only quintic bridge. Hidden values may be NaN and are never read."""
import numpy as np
from scipy.spatial.transform import Rotation
MATRIX=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,2,0,0,0],[1,1,1,1,1,1],[0,1,2,3,4,5],[0,0,2,6,12,20]],float)
def polynomial(values,gap):
    x=np.asarray(values,float);a,b=map(int,gap)
    if not 3<=a<b<=len(x)-3:raise ValueError('Three observed neighbors per side required')
    p=x[[a-3,a-2,a-1,b,b+1,b+2]];T=b-a+1
    if not np.isfinite(p).all():raise ValueError('Missing observed anchors')
    constraints=np.stack([p[2],(3*p[2]-4*p[1]+p[0])/2*T,(p[2]-2*p[1]+p[0])*T*T,p[3],(-3*p[3]+4*p[4]-p[5])/2*T,(p[5]-2*p[4]+p[3])*T*T])
    c=np.linalg.solve(MATRIX,constraints.reshape(6,-1));u=np.arange(1,b-a+1)/T
    result=x.copy();result[a:b]=(np.stack([u**i for i in range(6)],1)@c).reshape(b-a,*x.shape[1:]);return result
def observed_bridge(rotations,gap):
    R=np.asarray(rotations,float);a,b=map(int,gap)
    if R.shape[1:]!=(22,3,3) or a<3 or b>len(R)-3:raise ValueError('Body22 plus observed boundary support')
    ids=[a-3,a-2,a-1,b,b+1,b+2];anchor=R[a-1];relative=anchor[None].swapaxes(-1,-2)@R[ids]
    rv=Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(6,22,3)
    if np.linalg.norm(rv,axis=-1).max()>np.pi-.05:raise ValueError('Ambiguous SO3 chart')
    chart=np.full((len(R),22,3),np.nan);chart[ids]=rv;fill=polynomial(chart,gap)[a:b];out=R.copy()
    out[a:b]=anchor[None]@Rotation.from_rotvec(fill.reshape(-1,3)).as_matrix().reshape(b-a,22,3,3);return out
