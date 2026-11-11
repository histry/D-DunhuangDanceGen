"""EDGE CVPR2023 PFC, aligned to upstream eval/eval_pfc.py at 17c3428."""
import numpy as np
from dunhuang.contracts import Q
SOURCE='https://github.com/Stanford-TML/EDGE/blob/17c3428669ed6733edd9d8c66f7dc62060b8e46d/eval/eval_pfc.py'

def pfc(positions,fps=30,up_axis='y'):
    x=np.asarray(positions)
    if fps!=30:raise ValueError('Official EDGE PFC requires 30 FPS')
    if x.ndim!=3 or x.shape[1:]!=(24,3) or len(x)<3 or not np.isfinite(x).all():
        raise ValueError('Finite [T>=3,24,3] SMPL joints required')
    if up_axis not in ('y','z'):raise ValueError('Explicit Y/Z-up required')
    if up_axis=='y':x=x@Q.T
    dt=1/30
    velocity=(x[1:,0]-x[:-1,0])/dt
    acceleration=(velocity[1:]-velocity[:-1])/dt
    acceleration[:,2]=np.maximum(acceleration[:,2],0)
    acceleration=np.linalg.norm(acceleration,axis=-1)
    maximum=acceleration.max()
    if maximum==0:
        return dict(value=None,status='UNDEFINED_ZERO_ROOT_ACCELERATION',official_behavior='division by zero produces NaN',
            root_acceleration_max_mps2=0.,source=SOURCE,acceptance=None)
    acceleration/=maximum
    feet=x[:,[7,10,8,11]]
    displacement=np.linalg.norm(feet[2:,:,[0,1]]-feet[1:-1,:,[0,1]],axis=-1)
    minimum=np.zeros((len(displacement),2))
    minimum[:,0]=np.minimum(displacement[:,0],displacement[:,1])
    minimum[:,1]=np.minimum(displacement[:,2],displacement[:,3])
    score=(minimum[:,0]*minimum[:,1]*acceleration).mean()*10000
    return dict(value=float(score),status='COMPUTED',root_acceleration_max_mps2=float(maximum),
        source=SOURCE,units='official displacement product times 1e4; NOT m/s',fps=30,
        coordinates='Y-up converted by fixed proper Q to upstream Z-up',acceptance=None)
