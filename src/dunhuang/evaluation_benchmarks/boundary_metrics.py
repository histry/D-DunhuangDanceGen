import numpy as np

def boundaries(joints,indices,fps=30):
    x=np.asarray(joints);result=[]
    for t in sorted(set(int(i) for i in indices)):
        if not 2<=t<len(x)-1:continue
        velocity=np.diff(x[t-2:t+2],axis=0)*fps
        acceleration=np.diff(velocity,axis=0)*fps
        result.append(dict(frame=t,max_joint_step_m=float(np.linalg.norm(x[t]-x[t-1],axis=-1).max()),
            max_velocity_change_mps=float(np.linalg.norm(velocity[1]-velocity[0],axis=-1).max()),
            max_acceleration_jump_mps2=float(np.linalg.norm(acceleration[1]-acceleration[0],axis=-1).max())))
    return result
